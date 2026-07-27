import json
import sqlite3

import numpy as np

from acceleration_forecasting.retrieval.database import SCHEMA
from acceleration_forecasting.retrieval.guide_index import GuideIndex
from acceleration_forecasting.retrieval.guide_search import GuideSearchConfig


def add_candidate(
    connection, number, date, dataset, current, valid_months, available, embedding,
    *, direction="D", bin_start=2000.0,
):
    trend = f"t{number}"
    connection.execute(
        "INSERT INTO trends VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            trend, dataset, date, direction, bin_start, bin_start + 100.0, current,
            json.dumps([1.0] * 18), json.dumps([1] * valid_months + [0] * (18 - valid_months)),
            json.dumps([None] * 18), available, None, "", "",
        ),
    )
    vector = np.asarray(embedding, dtype=np.float32)
    connection.execute(
        "INSERT INTO waveform_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"r{number}", f"m{number}", date, direction, bin_start, bin_start + 100.0, 60.0,
            "source.csv", f"hash{number}", dataset, trend, vector.tobytes(), len(vector),
        ),
    )


def test_guide_filters_and_searches_past_rejected_candidates():
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA)
    # Rank 1 by similarity but invalid future-month count.
    add_candidate(connection, 1, "2020-01-01", "a", 1.0, 11, "2022-01-01", [1, 0])
    # Exactly +/- 0.5 is accepted.
    add_candidate(connection, 2, "2020-02-01", "b", 1.5, 12, "2022-01-01", [0.9, 0.1])
    # Outside current-value threshold.
    add_candidate(connection, 3, "2020-03-01", "c", 1.501, 18, "2022-01-01", [1, 0])
    # Future guide availability is rejected.
    add_candidate(connection, 4, "2020-04-01", "d", 1.0, 18, "2025-01-01", [1, 0])
    # Two waveforms on same day produce only one guide date.
    add_candidate(connection, 5, "2020-02-01", "e", 1.1, 18, "2022-01-01", [0.8, 0.2])
    index = GuideIndex(connection)
    result = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32), query_date="2024-01-01",
        query_dataset_id="query", query_current=1.0,
        allowed_dataset_ids={"a", "b", "c", "d", "e"},
        config=GuideSearchConfig(top_k=3),
    )
    assert len(result) == 1
    assert result[0]["measurement_date"] == "2020-02-01"
    assert abs(result[0]["current_max_difference"]) <= 0.5
    assert result[0]["valid_months"] >= 12


def test_hybrid_spatiotemporal_search_filters_near_future_but_allows_far_future():
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA)
    # Same dataset is always excluded, even when its guide is fully in the past.
    add_candidate(connection, 1, "2020-01-01", "query", 1.0, 18, "2022-01-01", [1, 0], bin_start=3300)
    # Same physical bin in another direction is allowed only when fully available in the past.
    add_candidate(connection, 2, "2020-02-01", "same-past", 1.0, 18, "2022-01-01", [0.9, 0.1], direction="U", bin_start=3300)
    add_candidate(connection, 3, "2020-03-01", "same-future", 1.0, 18, "2025-01-01", [1, 0], direction="U", bin_start=3300)
    # Adjacent 100 m follows the same temporal rule.
    add_candidate(connection, 4, "2020-04-01", "adjacent-past", 1.0, 18, "2022-01-01", [0.8, 0.2], bin_start=3400)
    add_candidate(connection, 5, "2020-05-01", "adjacent-future", 1.0, 18, "2025-01-01", [1, 0], bin_start=3200)
    # A 200 m-away guide may be used regardless of availability date.
    add_candidate(connection, 6, "2020-06-01", "far-future", 1.0, 18, "2025-01-01", [0.7, 0.3], bin_start=3500)
    index = GuideIndex(connection)
    config = GuideSearchConfig(
        top_k=5, strict_time=False, guide_search_mode="hybrid_spatiotemporal",
        near_distance_m=100.0, spatial_tolerance_m=1e-6,
        near_candidates_require_complete_past=True,
        far_candidates_strict_time=False, exclude_same_dataset=True,
    )
    result, diagnostics = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32), query_date="2024-01-01",
        query_dataset_id="query", query_current=1.0, query_bin_start_m=3300.0,
        allowed_dataset_ids={"query", "same-past", "same-future", "adjacent-past", "adjacent-future", "far-future"},
        config=config, return_diagnostics=True,
    )
    assert {item["dataset_id"] for item in result} == {
        "same-past", "adjacent-past", "far-future",
    }
    assert all(item["temporal_condition_applied"] for item in result if item["spatially_near"])
    assert next(item for item in result if item["dataset_id"] == "far-future")["temporal_condition_applied"] is False
    assert diagnostics["excluded_same_dataset"] == 1
    assert diagnostics["excluded_near_not_yet_available"] == 2


def test_hybrid_spatial_boundary_uses_tolerance():
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA)
    add_candidate(
        connection, 1, "2020-01-01", "boundary", 1.0, 18, "2025-01-01",
        [1, 0], bin_start=3400.0000005,
    )
    index = GuideIndex(connection)
    result = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32), query_date="2024-01-01",
        query_dataset_id="query", query_current=1.0, query_bin_start_m=3300.0,
        allowed_dataset_ids={"boundary"},
        config=GuideSearchConfig(
            strict_time=False, near_candidates_require_complete_past=True,
            far_candidates_strict_time=False,
        ),
    )
    assert result == []
