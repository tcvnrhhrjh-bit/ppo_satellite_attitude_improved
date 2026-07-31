# Improved PPO Satellite Attitude Control

This version keeps the original PPO structure but improves the project with
practical control-engineering checks, deterministic evaluation, and curriculum
learning for harder initial attitude errors.

## What changed

1. Reward shaping was corrected and expanded.
   - The reward includes attitude error, angular-rate penalty, control-effort penalty, and progress toward the target attitude.
   - The `progress_weight` parameter is now actually used instead of being ignored by a hard-coded constant.

2. The terminal condition is more realistic.
   - The original condition required the full state vector to be nearly identical to the target state.
   - The new condition succeeds when attitude error is below 5 degrees and angular-rate norm is below 0.01.

3. A PD controller baseline was added.
   - `baseline_pd.py` runs a simple quaternion PD controller.
   - This gives PPO a conventional control method to compare against.
   - The default PD gains were tuned to pass the 5 degree / 0.01 rad/s success condition in the standard 500-step test.

4. More analysis outputs were added.
   - Euler angle plot
   - Reward curve
   - Attitude-error curve
   - Angular-rate norm curve
   - Control-torque curve
   - Episode CSV history

5. PPO evaluation was made deterministic.
   - PPO still samples actions during training for exploration.
   - During testing, the policy now uses the highest-probability action instead of sampling randomly again.
   - This prevents a trained policy from looking like a random controller during the final demo.

6. Missing TensorFlow is handled gracefully.
   - If TensorFlow is unavailable, `main.py` explains the issue and runs the PD baseline instead of crashing.

7. Curriculum learning was added for PPO.
   - PPO no longer has to learn directly from a difficult 70 degree initial error.
   - The default training schedule starts from 10 degrees and gradually increases through 15, 20, 25, 30, 40, 50, 60, and 70 degrees.
   - This makes the learning problem easier at first, then progressively closer to the final mission scenario.

8. Reward shaping was strengthened for convergence.
   - Progress toward lower attitude error receives a larger reward.
   - Successful stabilization receives a larger terminal bonus.
   - Control-action penalty is reduced so PPO is less likely to learn an overly passive policy.

9. PD teacher warm-start was added.
   - Before PPO training, the actor can be briefly trained to imitate the PD baseline.
   - This gives the policy a useful initial direction instead of starting from fully random actions.
   - PPO then continues optimizing from that warmer starting point.

10. A faster RK4 dynamics integrator was added.
   - `rk4` is the default integrator for PPO rollouts because it is faster for repeated simulation steps.
   - `solve_ivp` remains available for comparison with an adaptive SciPy integrator.

## Main files

- `main.py`: improved entry point. Trains PPO, tests PPO, runs PD baseline, and writes comparison outputs.
- `torque_dynamics.py`: improved Gym environment, reward, done logic, and diagnostic tracking.
- `baseline_pd.py`: quaternion PD controller baseline.
- `utils.py`: plotting and CSV export helpers.
- `ppo_satellite_orientation.py`: original PPO model and hyperparameters.
- `residual_benchmark.py`: compares PD, PPO, and PD+PPO residual control under actuator nonlinearities and unknown disturbance torque.
- `residual_train.py`: trains PPO as a residual correction added on top of the PD controller.

## How to run

Run the verified PD baseline:

```bash
python main.py --pd-only --steps 500
```

Expected verified result:

```text
PD final attitude error: about 3.88 deg
PD final angular-rate norm: about 0.00998
PD success: True
```

Train and evaluate PPO when TensorFlow is installed:

```bash
python main.py --epochs 100
```

The default PPO training uses curriculum learning and a short PD teacher warm-start:

```bash
python main.py --epochs 300 --steps 500
```

A stronger Colab run can use:

```bash
python main.py --epochs 500 --steps 500 --steps-per-epoch 1000 --train-max-steps 120 --warm-start-epochs 30 --warm-start-lr 0.003 --integrator rk4
```

To train without curriculum learning:

```bash
python main.py --epochs 300 --steps 500 --no-curriculum --test-angle-deg 70
```

For a quick smoke test without training:

```bash
python main.py --skip-training --steps 200
```

The PD controller gains can be adjusted from the command line:

```bash
python main.py --pd-only --pd-kp 0.08 --pd-kd 0.80 --pd-torque-limit 0.1
```

Reward weights can also be adjusted:

```bash
python main.py --epochs 300 --progress-weight 8 --success-bonus 100 --action-weight 0.001
```

To compare integrators:

```bash
python main.py --pd-only --steps 500 --integrator rk4
python main.py --pd-only --steps 500 --integrator solve_ivp
```

Run the disturbed nonlinear benchmark:

```bash
python residual_benchmark.py --trials 30 --steps 500 --angles 20,30,40,50,60,70
```

Train the PD+PPO residual controller:

```bash
python residual_train.py --epochs 300 --steps-per-epoch 1000 --train-max-steps 120 --steps 500 --curriculum-angles 10,15,20,25,30,40,50,60,70
```

Then benchmark the saved residual policy:

```bash
python residual_benchmark.py --trials 30 --steps 500 --angles 20,30,40,50,60,70 --model residual_training_outputs/residual_model_weights.pickle
```

This benchmark compares:

```text
PD only
PPO only
PD + PPO residual correction
```

under:

```text
reaction-wheel dead zone
Coulomb and viscous friction
reduced actuator efficiency
unknown disturbance torque
```

Benchmark outputs include:

```text
benchmark_trials.csv
benchmark_summary.csv
benchmark_summary.pdf
controller diagnostic PDFs
controller episode-history CSVs
```

Outputs are written to:

```text
outputs_improved/
```

## How to explain this improvement

The original project demonstrated PPO attitude control in a simulation environment, but it lacked a strong engineering comparison and had a reward-design issue. This improved version corrects the reward, defines a practical success condition, adds a classical PD controller baseline, and introduces curriculum learning. The result is easier to evaluate because PPO can now be compared against a known control method using attitude error, angular rate, control effort, and total return.

The residual benchmark is designed to test a more realistic research hypothesis: PD is strong in the nominal case, but PD+PPO residual correction may become useful when the reaction-wheel actuator has dead zone, friction, reduced efficiency, and unknown disturbance torque. In this setup, PD provides the stabilizing baseline while PPO learns a residual correction rather than controlling the satellite from scratch.

The residual controller uses:

```text
u_total = u_PD + residual_scale * u_PPO
```

This means the PD controller remains responsible for basic stabilization, while PPO learns a small corrective term for nonlinear actuator effects and disturbances.

The most important implementation fix is that PPO evaluation is now deterministic. Training should use stochastic sampling, but final testing should use the best action selected by the learned policy. Without this change, the final PPO demo could fail simply because it was still sampling random actions.

## Suggested next step

Use the EnduroSat telemetry data as reset initial states. For example, convert TLM attitude angles and angular-rate telemetry into the 10-dimensional state vector, then sample real telemetry rows during `reset()`. That would connect the simulation task to actual satellite data.
