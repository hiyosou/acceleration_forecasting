from __future__ import annotations

import numpy as np
import pandas as pd


def select_monthly_history(group, anchor_date, months=3, search_days=15):
    anchor = pd.Timestamp(anchor_date).normalize()
    available = group.loc[group["measurement_date"] < anchor].copy()
    used = set()
    values = []
    masks = []
    selected = []
    anchor_month = anchor.replace(day=1)
    for offset in range(months, 0, -1):
        target = anchor_month - pd.DateOffset(months=offset)
        candidates = available.loc[
            (available["measurement_date"] - target).dt.days.abs() <= search_days
        ].copy()
        candidates = candidates.loc[~candidates["measurement_date"].isin(used)]
        if candidates.empty:
            values.append(np.nan)
            masks.append(0)
            selected.append("")
            continue
        candidates["_days"] = (candidates["measurement_date"] - target).dt.days.abs()
        candidates["_future_tie"] = (candidates["measurement_date"] > target).astype(int)
        row = candidates.sort_values(
            ["_days", "_future_tie", "measurement_date"], kind="mergesort"
        ).iloc[0]
        date = pd.Timestamp(row["measurement_date"]).normalize()
        used.add(date)
        value = float(row["current_acc_z_max"])
        values.append(value if np.isfinite(value) else np.nan)
        masks.append(int(np.isfinite(value)))
        selected.append(date.strftime("%Y-%m-%d"))
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=np.float32), selected


def sanitize_future(anchor_date, values, mask, cutoff_date, months=18):
    result = np.asarray([np.nan if value is None else value for value in values], dtype=np.float32)
    result_mask = np.asarray(mask, dtype=np.float32)
    if result.size != months or result_mask.size != months:
        raise ValueError(f"Future values and mask must contain {months} months")
    result_mask *= np.isfinite(result)
    cutoff = pd.to_datetime(cutoff_date, errors="coerce")
    if pd.notna(cutoff):
        first_month = pd.Timestamp(anchor_date).replace(day=1) + pd.DateOffset(months=1)
        for index in range(months):
            target_month = first_month + pd.DateOffset(months=index)
            if target_month >= pd.Timestamp(cutoff).normalize():
                result[index] = np.nan
                result_mask[index] = 0
    result[result_mask == 0] = np.nan
    return result, result_mask

