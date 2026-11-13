"""Non-learning baselines: random, reactive potential field, A* oracle."""
from __future__ import annotations

import heapq

import numpy as np


# --------------------------------------------------------------------- random
class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, env):
        return self.rng.uniform(-1.0, 1.0, size=(env.n, 3))


# ------------------------------------------------------------ potential field
class ReactivePolicy:
    """Classic artificial-potential-field controller driven by the raw lidar.

    Attractive force: unit vector towards the goal.
    Repulsive force: weighted sum of the directions of nearby lidar beams.
    The resulting desired velocity is tracked by the same low-level
    acceleration model as the learned policies.
    """

    def __init__(self, k_rep=1.0, safe_dist=1.5):
        self.k_rep = k_rep
        self.safe_dist = safe_dist

    def __call__(self, obs, env):
        R = env.n_rays
        lidar = np.asarray(obs[:, :R], np.float32) * env.perception_range
        g_dir = np.asarray(obs[:, R:R + 3], np.float32)
        vel = np.asarray(obs[:, R + 4:R + 7], np.float32) * env.v_max

        w = np.clip((self.safe_dist - lidar) / self.safe_dist, 0.0, 1.0) ** 2
        rep = -(w[:, :, None] * env.dirs[None, :, :]).sum(1)          # (n,2)
        rep /= (np.linalg.norm(rep, axis=1, keepdims=True) + 1e-6)
        attract = g_dir[:, :2] / (np.linalg.norm(g_dir[:, :2], axis=1, keepdims=True) + 1e-6)
        desired = attract + self.k_rep * rep
        desired /= (np.linalg.norm(desired, axis=1, keepdims=True) + 1e-6)

        v_des = np.zeros((env.n, 3), np.float32)
        v_des[:, :2] = desired * env.v_max
        v_des[:, 2] = g_dir[:, 2] * env.v_max
        act = (v_des - vel) / (env.a_max * env.dt)
        return np.clip(act, -1.0, 1.0)


# ---------------------------------------------------------------- A* (oracle)
class AStarPolicy:
    """Grid A* on the ground-truth map (privileged) + waypoint tracking."""

    def __init__(self, res=0.3, waypoint_tol=0.6, diagonal=True):
        self.res = res
        self.tol = waypoint_tol
        self.diagonal = diagonal

    # ---------------------------------------------------------------- planning
    def _astar(self, free, s, t):
        G = free.shape[0]
        if not free[s] or not free[t]:
            # snap to the closest free cell
            for p in (s, t):
                if not free[p]:
                    idx = np.argwhere(free)
                    d = np.abs(idx - np.array(p)).sum(1)
                    q = tuple(idx[d.argmin()])
                    if p == s:
                        s = q
                    else:
                        t = q
            if not free[s] or not free[t]:
                return None
        neighbours = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.diagonal:
            neighbours += [(-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        open_q = [(0.0, s)]
        g = {s: 0.0}
        parent = {}
        closed = set()
        h = lambda a: np.hypot(a[0] - t[0], a[1] - t[1])
        while open_q:
            _, cur = heapq.heappop(open_q)
            if cur == t:
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                return path[::-1]
            if cur in closed:
                continue
            closed.add(cur)
            for dx, dy, c in neighbours:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < G and 0 <= ny < G) or not free[nx, ny]:
                    continue
                ng = g[cur] + c
                if ng < g.get((nx, ny), 1e9):
                    g[(nx, ny)] = ng
                    parent[(nx, ny)] = cur
                    heapq.heappush(open_q, (ng + h((nx, ny)), (nx, ny)))
        return None

    def plan_batch(self, env):
        lim = env.map_half - env.drone_radius
        G = int(np.ceil(2 * lim / self.res)) + 1
        self.paths = []
        self.wp = []
        self.failed = 0
        for i in range(env.n):
            blocked = np.zeros((G, G), bool)
            xs = -lim + np.arange(G) * self.res
            X, Y = np.meshgrid(xs, xs, indexing="ij")
            for k in range(env.k):
                d = np.hypot(X - env.obs_c[i, k, 0], Y - env.obs_c[i, k, 1])
                blocked |= d < (env.obs_r[i, k] + env.drone_radius + 0.05)
            free = ~blocked
            s = (int(np.clip((env.start[i, 0] + lim) / self.res, 0, G - 1)),
                 int(np.clip((env.start[i, 1] + lim) / self.res, 0, G - 1)))
            t = (int(np.clip((env.goal[i, 0] + lim) / self.res, 0, G - 1)),
                 int(np.clip((env.goal[i, 1] + lim) / self.res, 0, G - 1)))
            path = self._astar(free, s, t)
            if path is None:
                self.paths.append(None)
                self.wp.append(0)
                self.failed += 1
            else:
                pts = np.array([(-lim + p[0] * self.res, -lim + p[1] * self.res) for p in path])
                self.paths.append(pts)
                self.wp.append(0)

    def reset(self, env):
        self.plan_batch(env)

    # ---------------------------------------------------------------- control
    def __call__(self, obs, env):
        act = np.zeros((env.n, 3), np.float32)
        for i in range(env.n):
            if not env.active[i] or self.paths[i] is None:
                continue
            pts = self.paths[i]
            while self.wp[i] < len(pts) - 1 and \
                    np.linalg.norm(pts[self.wp[i]] - env.pos[i, :2]) < self.tol:
                self.wp[i] += 1
            target = pts[self.wp[i]]
            d_vec = np.array([target[0], target[1], env.goal[i, 2]]) - env.pos[i]
            d = np.linalg.norm(d_vec)
            if self.wp[i] >= len(pts) - 1:          # final leg: aim at the goal
                d_vec = env.goal[i] - env.pos[i]
                d = np.linalg.norm(d_vec)
            v_des = env.v_max * d_vec / (d + 1e-6)
            act[i] = np.clip((v_des - env.vel[i]) / (env.a_max * env.dt), -1.0, 1.0)
        return act
