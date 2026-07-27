import json
import sqlite3

import numpy as np

from acceleration_forecasting.retrieval.database import SCHEMA
from acceleration_forecasting.retrieval.guide_index import GuideIndex
from acceleration_forecasting.retrieval.guide_search import GuideSearchConfig


def add_candidate(connection, number, date, dataset, current, valid_months, available, embedding):
    trend = f"t{number}"
    connection.execute(
        "INSERT INTO trends VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            trend, dataset, date, "D", 2000.0, 2100.0, current,
            json.dumps([1.0] * 18), json.dumps([1] * valid_months + [0] * (18 - valid_months)),
            json.dumps([None] * 18), available, None, "", "",
        ),
    )
    vector = np.asarray(embedding, dtype=np.float32)
    connection.execute(
        "INSERT INTO waveform_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"r{number}", f"m{number}", date, "D", 2000.0, 2100.0, 60.0,
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

