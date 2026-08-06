import argparse
import csv
import os
import subprocess
import sys


def run_command(command, cwd):
    print("\n$ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, field):
    value = row[field]
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def find_controller(rows, name):
    for row in rows:
        if row["controller"] == name:
            return row
    raise KeyError(f"Controller not found in benchmark summary: {name}")


def relative_to_report(path, target):
    return os.path.relpath(target, start=os.path.dirname(path)).replace("\\", "/")


def write_report(path, rows, train_dir, benchmark_dir, model_path, scenario, args):
    pd = find_controller(rows, "pd")
    residual = find_controller(rows, "pd_ppo_residual")
    ppo = find_controller(rows, "ppo")

    pd_success = as_float(pd, "success_rate")
    residual_success = as_float(residual, "success_rate")
    pd_error = as_float(pd, "mean_final_attitude_error_deg")
    residual_error = as_float(residual, "mean_final_attitude_error_deg")
    pd_energy = as_float(pd, "mean_control_energy")
    residual_energy = as_float(residual, "mean_control_energy")

    if residual_success > pd_success:
        verdict = "PD+PPO residual has a higher success rate than pure PD."
    elif residual_success == pd_success and residual_error < pd_error:
        verdict = "PD+PPO residual has the same success rate as pure PD and lower final attitude error."
    elif residual_success == pd_success and residual_energy < pd_energy:
        verdict = "PD+PPO residual has the same success rate as pure PD and lower control energy."
    else:
        verdict = "Pure PD is still better under this benchmark setting."

    lines = [
        "# Automated PD vs PD+PPO Residual Comparison",
        "",
        "## Scenario",
        "",
        f"`{scenario}`",
        "",
        "## Controller Parameters",
        "",
        f"- `pd_kp`: {args.pd_kp}",
        f"- `pd_kd`: {args.pd_kd}",
        f"- `pd_ki`: {args.pd_ki}",
        f"- `integral_limit`: {args.integral_limit}",
        f"- `residual_scale`: {args.residual_scale}",
        f"- `wheel_model`: {args.wheel_model}",
        f"- `wheel_torque_limit`: {args.wheel_torque_limit}",
        f"- `wheel_momentum_limit`: {args.wheel_momentum_limit}",
        f"- `wheel_power_limit`: {args.wheel_power_limit}",
        f"- `progress_weight`: {args.progress_weight}",
        f"- `near_target_weight`: {args.near_target_weight}",
        f"- `fine_pointing_weight`: {args.fine_pointing_weight}",
        f"- `regression_weight`: {args.regression_weight}",
        f"- `stagnation_weight`: {args.stagnation_weight}",
        f"- `adaptive_curriculum`: {args.adaptive_curriculum}",
        f"- `advance_return_threshold`: {args.advance_return_threshold}",
        f"- `advance_window`: {args.advance_window}",
        f"- `use_telemetry_reset`: {args.use_telemetry_reset}",
        f"- `telemetry_dir`: {args.telemetry_dir}",
        f"- `telemetry_gyro_unit`: {args.telemetry_gyro_unit}",
        f"- `telemetry_scalar_component`: {args.telemetry_scalar_component}",
        f"- `telemetry_eval_index`: {args.telemetry_eval_index}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Key Metrics",
        "",
        "| Controller | Success rate | Mean final error (deg) | Mean control energy |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["controller"]):
        lines.append(
            f"| {row['controller']} | {as_float(row, 'success_rate'):.3f} | "
            f"{as_float(row, 'mean_final_attitude_error_deg'):.3f} | "
            f"{as_float(row, 'mean_control_energy'):.6f} |"
        )

    if "mean_wheel_saturation_fraction" in rows[0]:
        lines.extend([
            "",
            "## Reaction Wheel Constraint Metrics",
            "",
            "| Controller | Mean wheel saturation fraction | Mean wheel power (W) |",
            "|---|---:|---:|",
        ])
        for row in sorted(rows, key=lambda r: r["controller"]):
            lines.append(
                f"| {row['controller']} | "
                f"{as_float(row, 'mean_wheel_saturation_fraction'):.3f} | "
                f"{as_float(row, 'mean_wheel_power_w'):.3f} |"
            )

    lines.extend([
        "",
        "## PD+PPO Residual Delta Versus Pure PD",
        "",
        f"- Success-rate delta: {residual_success - pd_success:+.3f}",
        f"- Final-error delta: {residual_error - pd_error:+.3f} deg",
        f"- Control-energy delta: {residual_energy - pd_energy:+.6f}",
        "",
        "Negative final-error and control-energy deltas are better for PD+PPO residual.",
        "",
        "## Output Files",
        "",
        f"- Training directory: `{relative_to_report(path, train_dir)}`",
        f"- Benchmark directory: `{relative_to_report(path, benchmark_dir)}`",
        f"- Residual model: `{relative_to_report(path, model_path)}`",
        f"- Benchmark summary CSV: `{relative_to_report(path, os.path.join(benchmark_dir, 'benchmark_summary.csv'))}`",
        f"- Benchmark trial CSV: `{relative_to_report(path, os.path.join(benchmark_dir, 'benchmark_trials.csv'))}`",
        f"- Benchmark summary plot: `{relative_to_report(path, os.path.join(benchmark_dir, 'benchmark_summary.pdf'))}`",
        "",
        "## Interpretation",
        "",
        "This pipeline tests whether PPO is useful as a residual correction on top of a classical PD controller.",
        "The residual policy is not expected to beat PD in every setting; it should be evaluated under nonlinear actuator behavior and disturbance torque.",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Automate PD+PPO residual training and pure-PD comparison.")
    parser.add_argument("--output-dir", default="automated_pd_residual_comparison")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--train-max-steps", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-angles", default="10,15,20,25,30,40,50,60,70")
    parser.add_argument("--benchmark-angles", default="20,30,40,50,60,70")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--pd-kp", type=float, default=0.08)
    parser.add_argument("--pd-kd", type=float, default=0.80)
    parser.add_argument("--pd-ki", type=float, default=0.0)
    parser.add_argument("--integral-limit", type=float, default=0.25)
    parser.add_argument("--pd-torque-limit", type=float, default=0.1)
    parser.add_argument("--wheel-model", choices=["none", "rw-0.01", "rw-0.03"], default="none")
    parser.add_argument("--wheel-torque-limit", type=float, default=None)
    parser.add_argument("--wheel-momentum-limit", type=float, default=None)
    parser.add_argument("--wheel-power-limit", type=float, default=None)
    parser.add_argument("--progress-weight", type=float, default=8.0)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--action-weight", type=float, default=0.001)
    parser.add_argument("--omega-weight", type=float, default=0.05)
    parser.add_argument("--near-target-weight", type=float, default=0.15)
    parser.add_argument("--fine-pointing-weight", type=float, default=0.40)
    parser.add_argument("--regression-weight", type=float, default=3.0)
    parser.add_argument("--stagnation-weight", type=float, default=0.03)
    parser.add_argument("--adaptive-curriculum", action="store_true")
    parser.add_argument("--advance-return-threshold", type=float, default=-150.0)
    parser.add_argument("--advance-window", type=int, default=5)
    parser.add_argument("--min-epochs-per-angle", type=int, default=8)
    parser.add_argument("--max-epochs-per-angle", type=int, default=50)
    parser.add_argument("--use-telemetry-reset", action="store_true")
    parser.add_argument("--telemetry-dir", default="")
    parser.add_argument("--telemetry-gyro-unit", choices=["deg/s", "rad/s"], default="deg/s")
    parser.add_argument("--telemetry-scalar-component", choices=["auto", "q0", "q1", "q2", "q3"], default="auto")
    parser.add_argument("--telemetry-eval-index", type=int, default=-1)
    parser.add_argument(
        "--scenario",
        choices=["nominal_nonlinear", "random_disturbance", "residual_friendly"],
        default="residual_friendly",
    )
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if args.use_telemetry_reset and not args.telemetry_dir:
        raise ValueError("--telemetry-dir is required when --use-telemetry-reset is enabled.")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir)
    train_dir = os.path.join(output_dir, "01_residual_training")
    benchmark_dir = os.path.join(output_dir, "02_benchmark")
    model_path = os.path.join(train_dir, "residual_model_weights.pickle")
    report_path = os.path.join(output_dir, "automated_comparison_report.md")

    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_training:
        train_command = [
            sys.executable,
            "residual_train.py",
            "--output-dir", train_dir,
            "--model-out", "residual_model_weights.pickle",
            "--epochs", str(args.epochs),
            "--steps-per-epoch", str(args.steps_per_epoch),
            "--train-max-steps", str(args.train_max_steps),
            "--steps", str(args.steps),
            "--curriculum-angles", args.train_angles,
            "--residual-scale", str(args.residual_scale),
            "--scenario", args.scenario,
            "--pd-kp", str(args.pd_kp),
            "--pd-kd", str(args.pd_kd),
            "--pd-ki", str(args.pd_ki),
            "--integral-limit", str(args.integral_limit),
            "--pd-torque-limit", str(args.pd_torque_limit),
            "--wheel-model", args.wheel_model,
            "--progress-weight", str(args.progress_weight),
            "--success-bonus", str(args.success_bonus),
            "--action-weight", str(args.action_weight),
            "--omega-weight", str(args.omega_weight),
            "--near-target-weight", str(args.near_target_weight),
            "--fine-pointing-weight", str(args.fine_pointing_weight),
            "--regression-weight", str(args.regression_weight),
            "--stagnation-weight", str(args.stagnation_weight),
            "--advance-return-threshold", str(args.advance_return_threshold),
            "--advance-window", str(args.advance_window),
            "--min-epochs-per-angle", str(args.min_epochs_per_angle),
            "--max-epochs-per-angle", str(args.max_epochs_per_angle),
        ]
        if args.wheel_torque_limit is not None:
            train_command.extend(["--wheel-torque-limit", str(args.wheel_torque_limit)])
        if args.wheel_momentum_limit is not None:
            train_command.extend(["--wheel-momentum-limit", str(args.wheel_momentum_limit)])
        if args.wheel_power_limit is not None:
            train_command.extend(["--wheel-power-limit", str(args.wheel_power_limit)])
        if args.adaptive_curriculum:
            train_command.append("--adaptive-curriculum")
        if args.use_telemetry_reset:
            train_command.extend([
                "--use-telemetry-reset",
                "--telemetry-dir", args.telemetry_dir,
                "--telemetry-gyro-unit", args.telemetry_gyro_unit,
                "--telemetry-scalar-component", args.telemetry_scalar_component,
                "--telemetry-eval-index", str(args.telemetry_eval_index),
            ])
        run_command(train_command, cwd=project_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Residual model not found: {model_path}")

    benchmark_command = [
        sys.executable,
        "residual_benchmark.py",
        "--output-dir", benchmark_dir,
        "--model", model_path,
        "--trials", str(args.trials),
        "--steps", str(args.steps),
        "--angles", args.benchmark_angles,
        "--residual-scale", str(args.residual_scale),
        "--scenario", args.scenario,
        "--pd-kp", str(args.pd_kp),
        "--pd-kd", str(args.pd_kd),
        "--pd-ki", str(args.pd_ki),
        "--integral-limit", str(args.integral_limit),
        "--pd-torque-limit", str(args.pd_torque_limit),
        "--wheel-model", args.wheel_model,
        "--progress-weight", str(args.progress_weight),
        "--success-bonus", str(args.success_bonus),
        "--action-weight", str(args.action_weight),
        "--omega-weight", str(args.omega_weight),
        "--near-target-weight", str(args.near_target_weight),
        "--fine-pointing-weight", str(args.fine_pointing_weight),
        "--regression-weight", str(args.regression_weight),
        "--stagnation-weight", str(args.stagnation_weight),
    ]
    if args.wheel_torque_limit is not None:
        benchmark_command.extend(["--wheel-torque-limit", str(args.wheel_torque_limit)])
    if args.wheel_momentum_limit is not None:
        benchmark_command.extend(["--wheel-momentum-limit", str(args.wheel_momentum_limit)])
    if args.wheel_power_limit is not None:
        benchmark_command.extend(["--wheel-power-limit", str(args.wheel_power_limit)])
    if args.use_telemetry_reset:
        benchmark_command.extend([
            "--use-telemetry-reset",
            "--telemetry-dir", args.telemetry_dir,
            "--telemetry-gyro-unit", args.telemetry_gyro_unit,
            "--telemetry-scalar-component", args.telemetry_scalar_component,
        ])
    run_command(benchmark_command, cwd=project_dir)

    summary_path = os.path.join(benchmark_dir, "benchmark_summary.csv")
    rows = read_summary(summary_path)
    write_report(report_path, rows, train_dir, benchmark_dir, model_path, args.scenario, args)
    print(f"\nAutomated comparison report written to: {report_path}")


if __name__ == "__main__":
    main()
