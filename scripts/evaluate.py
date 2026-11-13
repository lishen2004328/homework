"""Evaluate a trained policy (or a classical baseline) on fixed scenario sets.

python scripts/evaluate.py --kind ppo --model results/ppo_s0/model.pt \
       --conditions main,d10,d40,scale,noise05,noise15 --episodes 256 \
       --out results/ppo_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from uavnav.env import VecUAVNav  # noqa: E402
from uavnav.utils import evaluate_policy  # noqa: E402
from uavnav.baselines import RandomPolicy, ReactivePolicy, AStarPolicy  # noqa: E402


# Test conditions.  ``main`` matches the training distribution, the others
# probe generalisation along three axes (density, map scale, sensor noise).
CONDITIONS = {
    "main":   dict(map_half=7.5, n_obstacles=25),
    "d10":    dict(map_half=7.5, n_obstacles=10),
    "d40":    dict(map_half=7.5, n_obstacles=40),
    "scale":  dict(map_half=12.0, n_obstacles=64),
    "noise05": dict(map_half=7.5, n_obstacles=25, sensor_noise=0.05),
    "noise15": dict(map_half=7.5, n_obstacles=25, sensor_noise=0.15),
    "per4":   dict(map_half=7.5, n_obstacles=25, perception_range=4.0),
}


def build_env(cond_kwargs, n_envs, reward="shaped"):
    return VecUAVNav(n_envs=n_envs, seed=5, auto_reset=False, reward_mode=reward,
                     **cond_kwargs)


def build_policy(kind, model_path, env):
    if kind == "ppo":
        from uavnav.ppo import ActorCritic, policy_actions
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        net = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"],
                          ckpt["cfg"]["init_log_std"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return lambda obs, e: policy_actions(net, obs)
    if kind == "dqn":
        from uavnav.dqn import QNet, q_actions
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        net = QNet(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return lambda obs, e: e.DISCRETE_ACTIONS[q_actions(net, obs)]
    if kind == "random":
        return RandomPolicy(seed=0)
    if kind == "reactive":
        return ReactivePolicy(k_rep=1.0, safe_dist=1.5)
    if kind == "astar":
        return AStarPolicy(res=0.3)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["ppo", "dqn", "random", "reactive", "astar"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--conditions", default="main,d10,d40,scale,noise05,noise15")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = None
    results = {}
    for ci, name in enumerate(args.conditions.split(",")):
        env = build_env(CONDITIONS[name], args.n_envs)
        scenario = env.sample_scenario_set(args.episodes, seed=9300 + ci)
        if policy is None:
            policy = build_policy(args.kind, args.model, env)
        res = evaluate_policy(policy, env, scenario)
        if hasattr(policy, "failed"):
            res["astar_plan_failures"] = policy.failed
        results[name] = res
        print(f"[{args.kind}] {name:8s} succ {res['success']:.3f} coll {res['collision']:.3f} "
              f"oob {res['oob']:.3f} spl {res['spl']:.3f} path_ratio {res['path_ratio']:.3f}",
              flush=True)
    (out / f"eval_{args.kind}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
