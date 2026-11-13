"""Sanity checks + throughput benchmark for the simulator."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from uavnav.env import VecUAVNav

env = VecUAVNav(n_envs=64, seed=0)
obs = env.reset_all()
print("obs", obs.shape, "range", float(obs.min()), float(obs.max()))

# --- analytic lidar check -------------------------------------------------
e = VecUAVNav(n_envs=1, seed=1, n_obstacles=1, map_half=10.0, z_range=(2.0, 2.0))
e.start[0] = [0.0, 0.0, 2.0]
e.goal[0] = [5.0, 0.0, 2.0]
e.obs_c[0, 0] = [3.0, 0.0]
e.obs_r[0, 0] = 0.5
obs = e.load_batch(dict(start=e.start, goal=e.goal, obs_c=e.obs_c, obs_r=e.obs_r))
lidar = obs[0, :e.n_rays] * e.perception_range
ang = np.linspace(0, 2 * np.pi, e.n_rays, endpoint=False)
near = np.flatnonzero(lidar < 7.9)
print("ray0 dist %.3f (expect %.3f)" % (lidar[0], 3.0 - 0.75))
print("blocked rays/dirs", [(round(np.degrees(ang[i]), 1), round(float(lidar[i]), 2)) for i in near])

# --- rollout speed --------------------------------------------------------
env = VecUAVNav(n_envs=64, seed=0)
obs = env.reset_all()
rng = np.random.default_rng(0)
t0 = time.time(); steps = 400; n_ep = 0; last = None
for _ in range(steps):
    obs, rew, done, trunc, info = env.step(rng.uniform(-1, 1, size=(64, 3)))
    n_ep += len(info["episodes"])
    if info["episodes"]:
        last = info["episodes"][-1]
dt = time.time() - t0
print("env steps/s: %.0f  (%.2f ms per 64-env step)  episodes=%d" % (steps / dt, 1000 * dt / steps, n_ep))
print("episode sample:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in last.items() if k not in ("pos", "goal", "start")})
print("active all true in auto_reset mode:", bool(env.active.all()))
