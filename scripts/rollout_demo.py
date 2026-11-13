"""Record trajectories of a policy on a fixed set of scenarios (for figures).

python scripts/rollout_demo.py --kind ppo --model results/ppo_s0/model.pt \
       --out results/ppo_s0/traj.npz --n 8 --seed 9500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from uavnav.env import VecUAVNav  # noqa: E402
from scripts.evaluate import build_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["ppo", "dqn", "reactive", "astar", "random"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=9500)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    env = VecUAVNav(n_envs=args.n, seed=3, auto_reset=False)
    scen = env.sample_scenario_set(args.n, seed=args.seed)
    obs = env.load_batch(scen)
    policy = build_policy(args.kind, args.model, env)
    if hasattr(policy, "reset"):
        policy.reset(env)

    frames = []
    for _ in range(env.max_steps + 2):
        if not env.active.any():
            break
        frames.append(env.pos.copy())
        obs, _, _, _, _ = env.step(policy(obs, env))
    frames = np.stack(frames)                      # (T, n, 3)

    outcomes = {}
    # replay the final state: success == distance to goal below goal radius
    d = np.linalg.norm(env.goal - env.pos, axis=1)
    for i in range(args.n):
        outcomes[i] = dict(success=bool(d[i] <= env.goal_radius + 1e-3),
                           final_dist=float(d[i]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=frames, start=scen["start"], goal=scen["goal"],
             obs_c=scen["obs_c"], obs_r=scen["obs_r"],
             outcomes=np.array([outcomes[i]["success"] for i in range(args.n)]),
             map_half=env.map_half, kind=args.kind)
    succ = np.mean([outcomes[i]["success"] for i in range(args.n)])
    print(f"{args.kind}: {succ:.2f} success on the {args.n} demo scenarios -> {out}")


if __name__ == "__main__":
    main()
