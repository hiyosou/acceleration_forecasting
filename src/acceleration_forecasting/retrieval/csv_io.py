from __future__ import annotations

import pandas as pd


BASE_COLUMNS = [
    "updown", "seq", "distance[m]", "velocity[km/h]", "acc_x[m/s2]",
    "acc_y[m/s2]", "acc_z[m/s2]", "gyro_x[°/sec]", "gyro_y[°/sec]", "gyro_z[°/sec]",
]
OPTIONAL_COLUMNS = ["distance_corrected[m]", "position_correction_delta[m]"]


def load_vibration_csv(filepath):
    frame = None
    last_error = None
    for encoding in ("cp932", "utf-8-sig", "utf-8"):
        try:
            frame = pd.read_csv(filepath, header=None, encoding=encoding)
            break
        except Exception as error:
            last_error = error
    if frame is None:
        raise last_error
    if frame.empty or frame.shape[1] < len(BASE_COLUMNS):
        return None
    probe = str(frame.iloc[0, 2]).replace(".", "", 1).replace("-", "", 1)
    if not probe.isdigit():
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.empty:
        return None
    names = list(BASE_COLUMNS)
    for index in range(frame.shape[1] - len(BASE_COLUMNS)):
        names.append(OPTIONAL_COLUMNS[index] if index < len(OPTIONAL_COLUMNS) else f"extra_col_{index + 1}")
    frame = frame.iloc[:, : len(names)].copy()
    frame.columns = names
    numeric = names[1:]
    frame.loc[:, numeric] = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["distance[m]"]).reset_index(drop=True)

