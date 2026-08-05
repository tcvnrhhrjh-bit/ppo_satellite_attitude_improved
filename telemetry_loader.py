import csv
import glob
import os
from bisect import bisect_left

import numpy as np


QUAT_COLUMNS = [
    "iEstimatedORCquaternionQ0",
    "iEstimatedORCquaternionQ1",
    "iEstimatedORCquaternionQ2",
    "iEstimatedORCquaternionQ3",
]

GYRO_COLUMNS = [
    "fGYR0calibratedrateXcomponent",
    "fGYR0calibratedrateYcomponent",
    "fGYR0calibratedrateZcomponent",
]


def find_required_file(telemetry_dir, pattern):
    matches = sorted(glob.glob(os.path.join(telemetry_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No telemetry file matching {pattern!r} in {telemetry_dir}")
    return matches[0]


def parse_float(value):
    text = str(value).strip()
    if not text:
        return np.nan
    return float(text)


def read_quaternion_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                clock = int(float(row["clock"]))
                quat = np.array([parse_float(row[col]) for col in QUAT_COLUMNS], dtype=float)
            except (KeyError, ValueError):
                continue
            if np.all(np.isfinite(quat)):
                rows.append({"clock": clock, "quat_raw_order": quat})
    return rows


def first_column_indexes(header, names):
    indexes = []
    for name in names:
        try:
            indexes.append(header.index(name))
        except ValueError as exc:
            raise KeyError(f"Required telemetry column not found: {name}") from exc
    return indexes


def read_gyro_rows(path, gyro_unit):
    scale = 1.0
    if gyro_unit == "deg/s":
        scale = np.pi / 180.0
    elif gyro_unit != "rad/s":
        raise ValueError("gyro_unit must be 'deg/s' or 'rad/s'")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        clock_idx = header.index("clock")
        gyro_idx = first_column_indexes(header, GYRO_COLUMNS)
        for row in reader:
            try:
                clock = int(float(row[clock_idx]))
                omega = np.array([parse_float(row[idx]) for idx in gyro_idx], dtype=float) * scale
            except (IndexError, ValueError):
                continue
            if np.all(np.isfinite(omega)):
                rows.append({"clock": clock, "omega": omega})
    return rows


def infer_scalar_index(quat_rows):
    quats = np.asarray([row["quat_raw_order"] for row in quat_rows], dtype=float)
    return int(np.argmax(np.median(np.abs(quats), axis=0)))


def scalar_index_from_name(name, quat_rows):
    if name == "auto":
        return infer_scalar_index(quat_rows)
    mapping = {"q0": 0, "q1": 1, "q2": 2, "q3": 3}
    if name.lower() not in mapping:
        raise ValueError("scalar_component must be one of: auto, q0, q1, q2, q3")
    return mapping[name.lower()]


def reorder_quaternion_for_env(quat_raw_order, scalar_index):
    scalar = quat_raw_order[scalar_index]
    vector = [quat_raw_order[idx] for idx in range(4) if idx != scalar_index]
    quat = np.asarray([scalar] + vector, dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Invalid quaternion norm in telemetry row.")
    return quat / norm


def nearest_gyro(clock, gyro_rows, gyro_clocks):
    pos = bisect_left(gyro_clocks, clock)
    candidates = []
    if pos < len(gyro_rows):
        candidates.append(gyro_rows[pos])
    if pos > 0:
        candidates.append(gyro_rows[pos - 1])
    if not candidates:
        return np.zeros(3), None
    chosen = min(candidates, key=lambda row: abs(row["clock"] - clock))
    return chosen["omega"], chosen["clock"]


def load_telemetry_states(
    telemetry_dir,
    gyro_unit="deg/s",
    scalar_component="auto",
    max_time_offset_s=300.0,
):
    quat_path = find_required_file(telemetry_dir, "ADCS_Estimated_Quaternion_scaled*.csv")
    gyro_path = find_required_file(telemetry_dir, "analysis_scaled_8-panels*.csv")
    quat_rows = read_quaternion_rows(quat_path)
    gyro_rows = read_gyro_rows(gyro_path, gyro_unit=gyro_unit)
    if not quat_rows:
        raise ValueError(f"No usable quaternion rows found in {quat_path}")
    if not gyro_rows:
        raise ValueError(f"No usable gyro rows found in {gyro_path}")

    scalar_index = scalar_index_from_name(scalar_component, quat_rows)
    gyro_rows = sorted(gyro_rows, key=lambda row: row["clock"])
    gyro_clocks = [row["clock"] for row in gyro_rows]

    states = []
    records = []
    for row in quat_rows:
        omega, gyro_clock = nearest_gyro(row["clock"], gyro_rows, gyro_clocks)
        if gyro_clock is None or abs(gyro_clock - row["clock"]) > max_time_offset_s:
            continue
        quat = reorder_quaternion_for_env(row["quat_raw_order"], scalar_index)
        state = np.concatenate([quat, omega, np.zeros(3)])
        states.append(state)
        records.append({
            "clock": row["clock"],
            "gyro_clock": gyro_clock,
            "attitude_error_deg": 2.0 * np.rad2deg(np.arccos(np.clip(abs(quat[0]), -1.0, 1.0))),
            "omega_norm": float(np.linalg.norm(omega)),
        })

    if not states:
        raise ValueError("No synchronized quaternion and gyro telemetry states were created.")

    metadata = {
        "quat_path": quat_path,
        "gyro_path": gyro_path,
        "rows": len(states),
        "first_clock": records[0]["clock"],
        "last_clock": records[-1]["clock"],
        "scalar_component": f"q{scalar_index}",
        "gyro_unit": gyro_unit,
        "mean_attitude_error_deg": float(np.mean([record["attitude_error_deg"] for record in records])),
        "mean_omega_norm": float(np.mean([record["omega_norm"] for record in records])),
    }
    return np.asarray(states, dtype=float), records, metadata


def write_telemetry_state_summary(path, records, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["clock", "gyro_clock", "attitude_error_deg", "omega_norm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    meta_path = os.path.splitext(path)[0] + "_metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
