# Sensor-based UAV Navigation with PPO

Course project (深度学习课程设计, 2026 autumn).
**2026439189 李珅**

A lightweight, dependency-free-on-CPU benchmark for *end-to-end* UAV obstacle
avoidance, and a controlled study of three questions:

1. Can PPO learn a safe, efficient 3-D navigation policy from a 32-beam
   horizontal range sensor only?
2. How do **reward design** (potential-based shaping vs. sparse) and
   **sensing range** (8 m vs. 4 m) change sample efficiency and final performance?
3. Does the learned policy degrade gracefully under distribution shift
   (obstacle density, map scale, lidar noise)?

The whole study (12 training runs, 15 policies evaluated over 7 test
conditions) takes about 13 minutes on a 10-core desktop CPU: roughly 12
minutes for training in 6 parallel processes plus 32 s for the 105-row
evaluation.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Reproduce

```bash
# full matrix: 12 training runs + evaluation of 15 policies over 7 conditions
python scripts/run_suite.py --jobs 6

# or a single run
python scripts/train.py --algo ppo --seed 0 --steps 2000000 --out results/ppo_s0
python scripts/evaluate.py --kind ppo --model results/ppo_s0/model.pt \
       --conditions main,d10,d40,scale,noise05,noise15,per4 \
       --episodes 256 --out results/ppo_s0
python scripts/make_figures.py
python scripts/make_report.py      # rebuilds the PDF report from the measured data
```

## Layout

| Path | Content |
|---|---|
| `uavnav/env.py` | vectorised simulator (dynamics, lidar ray casting, rewards, episode bookkeeping) |
| `uavnav/ppo.py` | actor-critic + PPO/GAE trainer |
| `uavnav/dqn.py` | double DQN baseline on a 27-action discretisation |
| `uavnav/baselines.py` | random, artificial-potential-field, grid A* (privileged oracle) |
| `uavnav/utils.py` | metric aggregation (success / SPL / clearance ...) and fixed-scenario rollout |
| `scripts/` | training, evaluation, figures, experiment driver |
| `tests/test_env.py` | analytic lidar check + throughput benchmark |

## Task specification

* arena 15 × 15 m, altitude 1.0–3.5 m, 25 vertical cylindrical obstacles
  (radius 0.25–0.7 m), random start, goal 4.5–6.5 m away
* double-integrator dynamics: `a_max = 2 m/s²`, `v_max = 2 m/s`, `dt = 0.1 s`,
  150 steps per episode
* observation (48-D): 32 normalised lidar ranges, unit goal vector (3),
  goal distance (1), velocity (3), previous action (3), 4 lateral wall
  distances, 2 altitude margins
* action (3-D): acceleration command in [-1, 1]^3
* shaped reward: 2·Δ(progress) − 0.01 − 0.02‖a‖² − 0.25·max(0, 0.6−clearance)
  − 3 (collision / out of bounds) + 3 (success) − 0.5 (timeout)

## Results

See `results/summary.csv` (per run × condition) and `results/figures/`.
