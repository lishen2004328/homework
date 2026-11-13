"""Evaluation helpers shared by all algorithms."""
from __future__ import annotations

import numpy as np


def summarize(episodes):
    """Aggregate episode records into the metrics reported in the paper."""
    n = len(episodes)
    if n == 0:
        return dict(episodes=0)
    succ = np.array([e["success"] for e in episodes], float)
    coll = np.array([e["collision"] for e in episodes], float)
    oob = np.array([e["oob"] for e in episodes], float)
    tout = np.array([e["timeout"] for e in episodes], float)
    t = np.array([e["steps"] for e in episodes], float)
    path = np.array([e["path"] for e in episodes], float)
    d0 = np.array([e["dist0"] for e in episodes], float)
    g_r = np.array([e.get("goal_r", 0.0) for e in episodes], float)
    energy = np.array([e["energy"] for e in episodes], float)
    ret = np.array([e["ret"] for e in episodes], float)
    clear = np.array([e["min_clear"] for e in episodes], float)

    # success weighted by path length (Anderson et al., 2018);
    # the shortest feasible path ends at the goal radius, hence d0 - goal_r
    l_star = np.maximum(d0 - g_r, 0.1)
    spl = np.where(succ > 0, l_star / np.maximum(path, l_star), 0.0)
    ok = succ > 0.5
    return dict(
        episodes=n,
        success=float(succ.mean()), collision=float(coll.mean()),
        oob=float(oob.mean()), timeout=float(tout.mean()),
        spl=float(spl.mean()),
        steps=float(t[ok].mean()) if ok.any() else float("nan"),
        path=float(path[ok].mean()) if ok.any() else float("nan"),
        energy=float(energy.mean()), ret=float(ret.mean()),
        clearance_p05=float(np.percentile(clear, 5)),
        path_ratio=float((path[ok] / l_star[ok]).mean()) if ok.any() else float("nan"),
    )


def evaluate_policy(policy_fn, env, scenario):
    """Roll out ``policy_fn`` on a fixed scenario set.

    ``scenario`` must contain ``n`` episodes where ``n`` is a multiple of
    ``env.n``.  The environment is used in ``auto_reset=False`` mode so that
    no extra (unlabelled) episodes are generated.
    """
    n_scen = scenario["start"].shape[0]
    assert n_scen % env.n == 0, f"{n_scen} not divisible by {env.n}"
    episodes = []
    for b in range(0, n_scen, env.n):
        batch = {k: v[b:b + env.n] for k, v in scenario.items()}
        obs = env.load_batch(batch)
        if hasattr(policy_fn, "reset"):
            policy_fn.reset(env)
        for _ in range(env.max_steps + 2):
            if not env.active.any():
                break
            actions = policy_fn(obs, env)
            obs, _, _, _, info = env.step(actions)
            episodes.extend(info["episodes"])
    return summarize(episodes)
