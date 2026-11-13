"""Train PPO / DQN on the sensor-based UAV navigation task.

Example
-------
python scripts/train.py --algo ppo --seed 0 --steps 1000000 --out results/ppo_s0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from uavnav.env import VecUAVNav  # noqa: E402
from uavnav.ppo import PPOConfig, train_ppo  # noqa: E402
from uavnav.dqn import DQNConfig, train_dqn  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--map-half", type=float, default=7.5)
    p.add_argument("--n-obs", type=int, default=25)
    p.add_argument("--reward", choices=["shaped", "sparse"], default="shaped")
    p.add_argument("--perception", type=float, default=8.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-episodes", type=int, default=192)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-frames", action="store_true", default=False,
                   help="record high-frequency frames (not used during training)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    common = dict(map_half=args.map_half, n_obstacles=args.n_obs,
                  reward_mode=args.reward, perception_range=args.perception,
                  sensor_noise=args.noise)

    train_env = VecUAVNav(n_envs=args.n_envs, seed=1000 + args.seed, **common)
    eval_env = VecUAVNav(n_envs=32, seed=17 + args.seed, auto_reset=False, **common)
    scenario = eval_env.sample_scenario_set(args.eval_episodes, seed=9100 + args.seed)

    t0 = time.time()
    if args.algo == "ppo":
        cfg = PPOConfig(total_steps=args.steps, rollout_len=args.rollout_len,
                        seed=args.seed, ent_coef=args.ent_coef,
                        log_every=args.eval_every)
        if args.lr:
            cfg.lr = args.lr
        train_ppo(train_env, cfg, args.out, eval_env=eval_env, eval_scenario=scenario,
                  eval_every_updates=args.eval_every)
    else:
        cfg = DQNConfig(total_steps=args.steps, seed=args.seed,
                        eval_interval=args.eval_every)
        if args.lr:
            cfg.lr = args.lr
        train_dqn(train_env, cfg, args.out, eval_env=eval_env, eval_scenario=scenario)
    print(f"done in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
