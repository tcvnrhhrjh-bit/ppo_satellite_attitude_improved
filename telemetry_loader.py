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

HK_WHEEL_SPEED_COLUMNS = [
    "fRWL0measuredspeed",
    "fRWL1measuredspeed",
    "fRWL2measuredspeed",
    "fRWL3measuredspeed",
]

HK_WHEEL_COMMAND_COLUMNS = [
    "fRWL0openloopspeedcommand",
    "fRWL1openloopspeedcommand",
    "fRWL2openloopspeedcommand",
    "fRWL3openloopspeedcommand",
]

HK_ESTIMATED_TORQUE_COLUMNS = [
    "fEstimatedgyroscopictorqueXcomponent",
    "fEstimatedgyroscopictorqueYcomponent",
    "fEstimatedgyroscopictorqueZcomponent",
]

HK_MODE_COLUMNS = ["eOrbitmode", "eADCSrunmode", "eControlmode"]


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


def nearest_row(clock, rows, clocks):
    pos = bisect_left(clocks, clock)
    candidates = []
    if pos < len(rows):
        candidates.append(rows[pos])
    if pos > 0:
        candidates.append(rows[pos - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(row["clock"] - clock))


def load_housekeeping_rows(
    telemetry_dir,
    wheel_momentum_limit=0.04,
    wheel_speed_reference=6500.0,
    disturbance_scale=1.0,
):
    hk_path = find_required_file(telemetry_dir, "*RIoT-2-hk.csv")
    rows = []
    with open(hk_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        clock_idx = header.index("clock")
        wheel_speed_idx = first_column_indexes(header, HK_WHEEL_SPEED_COLUMNS)
        wheel_command_idx = first_column_indexes(header, HK_WHEEL_COMMAND_COLUMNS)
        torque_idx = first_column_indexes(header, HK_ESTIMATED_TORQUE_COLUMNS)
        mode_idx = first_column_indexes(header, HK_MODE_COLUMNS)
        for row in reader:
            try:
                clock = int(float(row[clock_idx]))
                wheel_speed = np.array([parse_float(row[idx]) for idx in wheel_speed_idx], dtype=float)
                wheel_command = np.array([parse_float(row[idx]) for idx in wheel_command_idx], dtype=float)
                estimated_torque = np.array([parse_float(row[idx]) for idx in torque_idx], dtype=float) * disturbance_scale
                modes = [row[idx] for idx in mode_idx]
            except (IndexError, ValueError):
                continue
            if not np.all(np.isfinite(wheel_speed[:3])):
                continue
            wheel_momentum = np.clip(
                wheel_speed[:3] / wheel_speed_reference,
                -1.0,
                1.0,
            ) * wheel_momentum_limit
            if not np.all(np.isfinite(estimated_torque)):
                estimated_torque = np.zeros(3)
            rows.append({
                "clock": clock,
                "wheel_speed": wheel_speed,
                "wheel_command": wheel_command,
                "wheel_momentum": wheel_momentum,
                "estimated_torque": estimated_torque,
                "orbit_mode": modes[0],
                "adcs_run_mode": modes[1],
                "control_mode": modes[2],
            })
    if not rows:
        raise ValueError(f"No usable housekeeping rows found in {hk_path}")
    return sorted(rows, key=lambda row: row["clock"]), hk_path


def load_telemetry_states(
    telemetry_dir,
    gyro_unit="deg/s",
    scalar_component="auto",
    max_time_offset_s=300.0,
    use_housekeeping=False,
    housekeeping_dir=None,
    housekeeping_wheel_momentum_limit=0.04,
    housekeeping_wheel_speed_reference=6500.0,
    housekeeping_disturbance_scale=1.0,
    telemetry_min_initial_error=None,
    telemetry_max_initial_error=None,
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
    hk_rows = []
    hk_clocks = []
    hk_path = ""
    if use_housekeeping:
        hk_rows, hk_path = load_housekeeping_rows(
            housekeeping_dir or telemetry_dir,
            wheel_momentum_limit=housekeeping_wheel_momentum_limit,
            wheel_speed_reference=housekeeping_wheel_speed_reference,
            disturbance_scale=housekeeping_disturbance_scale,
        )
        hk_clocks = [row["clock"] for row in hk_rows]

    states = []
    records = []
    disturbances = []
    for row in quat_rows:
        omega, gyro_clock = nearest_gyro(row["clock"], gyro_rows, gyro_clocks)
        if gyro_clock is None or abs(gyro_clock - row["clock"]) > max_time_offset_s:
            continue
        quat = reorder_quaternion_for_env(row["quat_raw_order"], scalar_index)
        wheel_momentum = np.zeros(3)
        hk_clock = None
        hk_record = None
        if use_housekeeping:
            hk_record = nearest_row(row["clock"], hk_rows, hk_clocks)
            if hk_record is not None and abs(hk_record["clock"] - row["clock"]) <= max_time_offset_s:
                wheel_momentum = hk_record["wheel_momentum"]
                hk_clock = hk_record["clock"]
        attitude_error_deg = 2.0 * np.rad2deg(np.arccos(np.clip(abs(quat[0]), -1.0, 1.0)))
        if telemetry_min_initial_error is not None and attitude_error_deg < telemetry_min_initial_error:
            continue
        if telemetry_max_initial_error is not None and attitude_error_deg > telemetry_max_initial_error:
            continue
        state = np.concatenate([quat, omega, wheel_momentum])
        states.append(state)
        disturbance = hk_record["estimated_torque"] if hk_record is not None else np.zeros(3)
        disturbances.append(disturbance)
        wheel_speed = hk_record["wheel_speed"] if hk_record is not None else np.full(4, np.nan)
        wheel_command = hk_record["wheel_command"] if hk_record is not None else np.full(4, np.nan)
        records.append({
            "clock": row["clock"],
            "gyro_clock": gyro_clock,
            "hk_clock": hk_clock,
            "attitude_error_deg": float(attitude_error_deg),
            "omega_norm": float(np.linalg.norm(omega)),
            "wheel_momentum_x": float(wheel_momentum[0]),
            "wheel_momentum_y": float(wheel_momentum[1]),
            "wheel_momentum_z": float(wheel_momentum[2]),
            "rwl0_measured_speed": float(wheel_speed[0]),
            "rwl1_measured_speed": float(wheel_speed[1]),
            "rwl2_measured_speed": float(wheel_speed[2]),
            "rwl3_measured_speed": float(wheel_speed[3]),
            "rwl0_speed_command": float(wheel_command[0]),
            "rwl1_speed_command": float(wheel_command[1]),
            "rwl2_speed_command": float(wheel_command[2]),
            "rwl3_speed_command": float(wheel_command[3]),
            "hk_disturbance_x": float(disturbance[0]),
            "hk_disturbance_y": float(disturbance[1]),
            "hk_disturbance_z": float(disturbance[2]),
            "orbit_mode": hk_record["orbit_mode"] if hk_record is not None else "",
            "adcs_run_mode": hk_record["adcs_run_mode"] if hk_record is not None else "",
            "control_mode": hk_record["control_mode"] if hk_record is not None else "",
        })

    if not states:
        raise ValueError(
            "No synchronized quaternion and gyro telemetry states matched the requested filters. "
            f"min_initial_error={telemetry_min_initial_error}, "
            f"max_initial_error={telemetry_max_initial_error}."
        )

    metadata = {
        "quat_path": quat_path,
        "gyro_path": gyro_path,
        "housekeeping_path": hk_path,
        "use_housekeeping": use_housekeeping,
        "rows": len(states),
        "first_clock": records[0]["clock"],
        "last_clock": records[-1]["clock"],
        "scalar_component": f"q{scalar_index}",
        "gyro_unit": gyro_unit,
        "telemetry_min_initial_error": telemetry_min_initial_error,
        "telemetry_max_initial_error": telemetry_max_initial_error,
        "min_attitude_error_deg": float(np.min([record["attitude_error_deg"] for record in records])),
        "max_attitude_error_deg": float(np.max([record["attitude_error_deg"] for record in records])),
        "filtered_out_rows": len(quat_rows) - len(states),
        "mean_attitude_error_deg": float(np.mean([record["attitude_error_deg"] for record in records])),
        "mean_omega_norm": float(np.mean([record["omega_norm"] for record in records])),
        "mean_wheel_momentum_norm": float(np.mean([
            np.linalg.norm([record["wheel_momentum_x"], record["wheel_momentum_y"], record["wheel_momentum_z"]])
            for record in records
        ])),
        "mean_hk_disturbance_norm": float(np.mean([np.linalg.norm(item) for item in disturbances])),
    }
    return np.asarray(states, dtype=float), records, metadata, np.asarray(disturbances, dtype=float)


def write_telemetry_state_summary(path, records, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clock", "gyro_clock", "hk_clock", "attitude_error_deg", "omega_norm",
            "wheel_momentum_x", "wheel_momentum_y", "wheel_momentum_z",
            "rwl0_measured_speed", "rwl1_measured_speed", "rwl2_measured_speed", "rwl3_measured_speed",
            "rwl0_speed_command", "rwl1_speed_command", "rwl2_speed_command", "rwl3_speed_command",
            "hk_disturbance_x", "hk_disturbance_y", "hk_disturbance_z",
            "orbit_mode", "adcs_run_mode", "control_mode",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    meta_path = os.path.splitext(path)[0] + "_metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

