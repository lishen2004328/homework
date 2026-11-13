"""Lightweight vectorised 3-D UAV navigation simulator (NumPy only).

A simplified quadrotor is modelled as a point mass with second-order
dynamics (acceleration commands).  Obstacles are vertical cylinders that
span the whole altitude band, so a horizontal 360-degree range sensor is
sufficient to perceive the scene.  The setting preserves the properties
that make sensor-based navigation hard: partial observability, non-convex
obstacles and continuous 3-D control, while remaining cheap enough to
train millions of environment steps on a laptop CPU.
"""

from __future__ import annotations

import numpy as np


def _sample_one(rng, i, start, goal, obs_c, obs_r, k, map_half,
                r_range, z_range, goal_dist, drone_radius):
    """Fill start/goal/obstacles of scenario ``i`` in place."""
    lim = map_half - 2.0 * drone_radius
    start[i, 0] = rng.uniform(-lim, lim)
    start[i, 1] = rng.uniform(-lim, lim)
    start[i, 2] = rng.uniform(*z_range)
    # goal: random bearing, distance in ``goal_dist``
    ang = rng.uniform(0.0, 2.0 * np.pi)
    rad = rng.uniform(*goal_dist)
    goal[i, 0] = np.clip(start[i, 0] + rad * np.cos(ang), -lim, lim)
    goal[i, 1] = np.clip(start[i, 1] + rad * np.sin(ang), -lim, lim)
    goal[i, 2] = rng.uniform(*z_range)

    placed = 0
    for _ in range(80 * k):
        if placed >= k:
            break
        c = rng.uniform(-lim, lim, size=2)
        r = rng.uniform(*r_range)
        if np.linalg.norm(c - start[i, :2]) < r + drone_radius + 0.8:
            continue
        if np.linalg.norm(c - goal[i, :2]) < r + drone_radius + 0.8:
            continue
        if placed > 0:
            gaps = np.linalg.norm(obs_c[i, :placed] - c, axis=1) - obs_r[i, :placed] - r
            if gaps.min() < 0.6:      # keep corridors passable
                continue
        obs_c[i, placed] = c
        obs_r[i, placed] = r
        placed += 1
    # if the rejection sampler stalls, relax the corridor constraint
    while placed < k:
        c = rng.uniform(-lim, lim, size=2)
        r = rng.uniform(*r_range)
        if (np.linalg.norm(c - start[i, :2]) < r + drone_radius + 0.5 or
                np.linalg.norm(c - goal[i, :2]) < r + drone_radius + 0.5):
            continue
        obs_c[i, placed] = c
        obs_r[i, placed] = r
        placed += 1


class VecUAVNav:
    """Vectorised navigation task.

    Observation (``obs_dim`` = ``n_rays`` + 16):
        [0 : n_rays]                 horizontal range readings in [0, 1]
                                     (0 = obstacle touching, 1 = free up to
                                     ``perception_range``)
        [n_rays : n_rays+3]          unit vector towards the goal (world frame)
        [n_rays+3]                   goal distance / 10 m
        [n_rays+4 : n_rays+7]        velocity / v_max
        [n_rays+7 : n_rays+10]       previous acceleration command
        [n_rays+10 : n_rays+14]      distances to the four lateral walls, / lim
        [n_rays+14 : n_rays+16]      distances to the lower/upper altitude
                                     limits, / 2 m
    Action: 3-D acceleration command in [-1, 1]^3 (scaled by ``a_max``).
    """

    # discrete action set used by DQN: 3^3 = 27 combinations of {-1,0,1}
    DISCRETE_ACTIONS = np.array([[x, y, z] for x in (-1.0, 0.0, 1.0)
                                 for y in (-1.0, 0.0, 1.0)
                                 for z in (-1.0, 0.0, 1.0)], dtype=np.float32)

    def __init__(self, n_envs=64, seed=0, map_half=7.5, n_obstacles=25,
                 obstacle_radius=(0.25, 0.70), n_rays=32, max_range=8.0,
                 perception_range=8.0, v_max=2.0, a_max=2.0, dt=0.1,
                 max_steps=150, z_range=(1.0, 3.5), drone_radius=0.25,
                 reward_mode="shaped", sensor_noise=0.0, goal_radius=0.5,
                 goal_dist=(4.5, 6.5), auto_reset=True):
        self.n = int(n_envs)
        self.rng = np.random.default_rng(seed)
        self.map_half = float(map_half)
        self.k = int(n_obstacles)
        self.r_range = obstacle_radius
        self.n_rays = int(n_rays)
        self.max_range = float(max_range)
        self.perception_range = float(perception_range)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.z_range = z_range
        self.drone_radius = float(drone_radius)
        self.reward_mode = reward_mode
        self.sensor_noise = float(sensor_noise)
        self.goal_radius = float(goal_radius)
        self.goal_dist = goal_dist
        self.auto_reset = bool(auto_reset)

        angles = np.linspace(0.0, 2.0 * np.pi, self.n_rays, endpoint=False)
        self.dirs = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)
        self.dist_scale = 10.0

        self.pos = np.zeros((self.n, 3), np.float32)
        self.vel = np.zeros((self.n, 3), np.float32)
        self.goal = np.zeros((self.n, 3), np.float32)
        self.start = np.zeros((self.n, 3), np.float32)
        self.obs_c = np.zeros((self.n, self.k, 2), np.float32)
        self.obs_r = np.zeros((self.n, self.k), np.float32)
        self.obs_R = np.zeros((self.n, self.k), np.float32)
        self.action = np.zeros((self.n, 3), np.float32)
        self.t = np.zeros(self.n, np.float32)
        self.path = np.zeros(self.n, np.float32)
        self.energy = np.zeros(self.n, np.float32)
        self.ret = np.zeros(self.n, np.float32)
        self.min_clear = np.full(self.n, 10.0, np.float32)
        self.prev_dist = np.zeros(self.n, np.float32)
        self.active = np.ones(self.n, bool)
        self.config = dict(n_envs=self.n, map_half=self.map_half, n_obstacles=self.k,
                           obstacle_radius=list(self.r_range), n_rays=self.n_rays,
                           perception_range=self.perception_range, v_max=self.v_max,
                           a_max=self.a_max, dt=self.dt, max_steps=self.max_steps,
                           z_range=list(self.z_range), reward_mode=self.reward_mode,
                           sensor_noise=self.sensor_noise, goal_radius=self.goal_radius,
                           goal_dist=list(self.goal_dist), obs_dim=self.obs_dim)
        self.reset_all()

    # ------------------------------------------------------------------ utils
    @property
    def obs_dim(self):
        return self.n_rays + 16

    @property
    def act_dim(self):
        return 3

    def _refresh_obs_R(self):
        self.obs_R[:] = self.obs_r + self.drone_radius

    def _goal_dist(self):
        return np.linalg.norm(self.goal - self.pos, axis=1) + 1e-6

    # ----------------------------------------------------------- observation
    def _lidar(self, pos, obs_c, obs_R):
        rel = obs_c - pos[:, None, :2]                      # (M,K,2)
        b = rel @ self.dirs.T                              # (M,K,R)
        c2 = (rel ** 2).sum(-1) - obs_R ** 2                # (M,K)
        disc = b * b - c2[:, :, None]
        t_near = b - np.sqrt(np.maximum(disc, 0.0))
        # a beam only hits when it actually points into the obstacle
        valid = (disc > 0) & (b > 0) & (t_near > 0)
        t = np.where(valid, t_near, np.inf)
        dist = t.min(axis=1)                                # (M,R)
        inside = c2.min(axis=1) < 0.0                       # origin inside an obstacle
        dist = np.where(inside[:, None], 0.0, dist)
        dist = np.minimum(dist, self.perception_range)
        if self.sensor_noise > 0.0:
            dist = dist + self.rng.normal(0.0, self.sensor_noise * self.perception_range,
                                          size=dist.shape)
        return np.clip(dist, 0.0, self.perception_range)

    def _build_obs(self, pos, goal, vel, action, obs_c, obs_R):
        d_vec = goal - pos
        d = np.linalg.norm(d_vec, axis=1, keepdims=True) + 1e-6
        g_dir = d_vec / d
        lidar = self._lidar(pos, obs_c, obs_R) / self.perception_range
        lim = self.map_half - self.drone_radius
        zlo = self.z_range[0] - 0.5
        zhi = self.z_range[1] + 0.5
        walls = np.stack([
            (lim - pos[:, 0]) / (2.0 * lim),
            (lim + pos[:, 0]) / (2.0 * lim),
            (lim - pos[:, 1]) / (2.0 * lim),
            (lim + pos[:, 1]) / (2.0 * lim),
            (pos[:, 2] - zlo) / (zhi - zlo),
            (zhi - pos[:, 2]) / (zhi - zlo),
        ], axis=1)
        obs = np.concatenate(
            [lidar.astype(np.float32), g_dir.astype(np.float32),
             (d / self.dist_scale).astype(np.float32),
             (vel / self.v_max).astype(np.float32),
             action.astype(np.float32),
             np.clip(walls, 0.0, 1.0).astype(np.float32)], axis=1)
        return obs.astype(np.float32)

    # ---------------------------------------------------------------- resets
    def _reset_one(self, i):
        _sample_one(self.rng, i, self.start, self.goal, self.obs_c, self.obs_r,
                    self.k, self.map_half, self.r_range, self.z_range,
                    self.goal_dist, self.drone_radius)
        self.obs_R[i] = self.obs_r[i] + self.drone_radius
        self.pos[i] = self.start[i]
        self.vel[i] = 0.0
        self.action[i] = 0.0
        self.t[i] = 0.0
        self.path[i] = 0.0
        self.energy[i] = 0.0
        self.ret[i] = 0.0
        self.min_clear[i] = 10.0
        self.prev_dist[i] = self._goal_dist()[i]
        self.active[i] = True

    def reset_all(self):
        for i in range(self.n):
            self._reset_one(i)
        return self._build_obs(self.pos, self.goal, self.vel, self.action,
                               self.obs_c, self.obs_R)

    # --------------------------------------------------- external scenarios
    def sample_scenario_set(self, n, seed):
        """Generate a scenario set with an independent RNG (shared eval maps)."""
        rng = np.random.default_rng(seed)
        s = dict(start=np.zeros((n, 3), np.float32),
                 goal=np.zeros((n, 3), np.float32),
                 obs_c=np.zeros((n, self.k, 2), np.float32),
                 obs_r=np.zeros((n, self.k), np.float32))
        for i in range(n):
            _sample_one(rng, i, s["start"], s["goal"], s["obs_c"], s["obs_r"],
                        self.k, self.map_half, self.r_range, self.z_range,
                        self.goal_dist, self.drone_radius)
        return s

    def load_batch(self, scenario):
        """Load exactly ``n_envs`` scenarios (one per environment)."""
        m = scenario["start"].shape[0]
        assert m == self.n, f"batch size {m} != n_envs {self.n}"
        self.start[:] = scenario["start"]
        self.goal[:] = scenario["goal"]
        self.obs_c[:] = scenario["obs_c"]
        self.obs_r[:] = scenario["obs_r"]
        self._refresh_obs_R()
        self.pos[:] = self.start
        self.vel[:] = 0.0
        self.action[:] = 0.0
        self.t[:] = 0.0
        self.path[:] = 0.0
        self.energy[:] = 0.0
        self.ret[:] = 0.0
        self.min_clear[:] = 10.0
        self.active[:] = True
        self.prev_dist[:] = self._goal_dist()
        return self._build_obs(self.pos, self.goal, self.vel, self.action,
                               self.obs_c, self.obs_R)

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, np.float32).reshape(self.n, 3), -1.0, 1.0)
        act = self.active.copy()
        self.action[act] = a[act]

        # ---- dynamics (double integrator, masked by ``active``)
        acc = np.where(self.active[:, None], a * self.a_max, 0.0).astype(np.float32)
        self.vel = np.where(self.active[:, None],
                            np.clip(self.vel + acc * self.dt, -self.v_max, self.v_max),
                            self.vel).astype(np.float32)
        self.pos = np.where(self.active[:, None],
                            self.pos + self.vel * self.dt, self.pos).astype(np.float32)
        self.t += self.active.astype(np.float32)
        self.path += np.where(self.active, np.linalg.norm(self.vel, axis=1) * self.dt, 0.0)
        self.energy += np.where(self.active, (a ** 2).sum(1) * self.dt, 0.0)

        # ---- geometry
        dist = self._goal_dist()
        progress = self.prev_dist - dist
        rel = self.obs_c - self.pos[:, None, :2]
        d_centers = np.linalg.norm(rel, axis=2)                      # (N,K)
        clearance = d_centers - self.obs_R
        min_clear = clearance.min(axis=1)
        collision = (min_clear < 0.0) & self.active
        self.min_clear = np.where(self.active, np.minimum(self.min_clear, min_clear),
                                  self.min_clear)

        lim = self.map_half - self.drone_radius
        oob = ((np.abs(self.pos[:, 0]) > lim) | (np.abs(self.pos[:, 1]) > lim) |
               (self.pos[:, 2] < self.z_range[0] - 0.5) |
               (self.pos[:, 2] > self.z_range[1] + 0.5)) & self.active
        success = (dist <= self.goal_radius) & self.active
        timeout = (self.t >= self.max_steps) & self.active & ~success & ~collision & ~oob

        # ---- reward
        if self.reward_mode == "shaped":
            rew = (2.0 * progress - 0.01
                   - 0.02 * (a ** 2).sum(1)
                   - 0.25 * np.maximum(0.0, 0.6 - min_clear))
        elif self.reward_mode == "sparse":
            rew = -0.01 * np.ones(self.n, np.float32)
        else:
            raise ValueError(self.reward_mode)
        rew = rew + 3.0 * success - 3.0 * (collision | oob) - 0.5 * timeout
        rew = np.where(self.active, rew, 0.0).astype(np.float32)
        self.ret += rew
        self.prev_dist = np.where(self.active, dist, self.prev_dist)

        fin = success | collision | oob | timeout
        obs = self._build_obs(self.pos, self.goal, self.vel, self.action,
                              self.obs_c, self.obs_R)
        obs_pre = obs.copy()

        episodes = []
        for i in np.flatnonzero(fin):
            episodes.append(dict(
                idx=int(i),
                success=bool(success[i]), collision=bool(collision[i]),
                oob=bool(oob[i]), timeout=bool(timeout[i]),
                time=float(self.t[i]), ret=float(self.ret[i]),
                steps=float(self.t[i]),
                goal_r=float(self.goal_radius),
                path=float(self.path[i]), energy=float(self.energy[i]),
                dist0=float(np.linalg.norm(self.start[i] - self.goal[i])),
                min_clear=float(self.min_clear[i]),
                pos=self.pos[i].copy(), goal=self.goal[i].copy(),
                start=self.start[i].copy(),
            ))

        if self.auto_reset:
            for i in np.flatnonzero(fin):
                self._reset_one(i)
            if fin.any():
                obs[fin] = self._build_obs(self.pos[fin], self.goal[fin],
                                           self.vel[fin], self.action[fin],
                                           self.obs_c[fin], self.obs_R[fin])
        else:
            self.active[fin] = False

        info = dict(episodes=episodes,
                    trunc=np.flatnonzero(timeout),
                    obs_pre=obs_pre,
                    done=fin)
        return obs, rew, fin, timeout, info
