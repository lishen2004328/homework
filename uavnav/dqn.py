"""Double DQN baseline with a discretised 27-action space."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap, obs_dim, seed=0):
        self.cap = cap
        self.o = np.empty((cap, obs_dim), np.float32)
        self.a = np.empty((cap,), np.int64)
        self.r = np.empty((cap,), np.float32)
        self.o2 = np.empty((cap, obs_dim), np.float32)
        self.d = np.empty((cap,), np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, o2, d):
        n = len(a)
        idx = (self.ptr + np.arange(n)) % self.cap
        self.o[idx] = o
        self.a[idx] = a
        self.r[idx] = r
        self.o2[idx] = o2
        self.d[idx] = d
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, batch)
        return (torch.as_tensor(self.o[idx]), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(self.o2[idx]),
                torch.as_tensor(self.d[idx]))


@dataclass
class DQNConfig:
    total_steps: int = 1_000_000
    gamma: float = 0.99
    lr: float = 1e-4
    batch: int = 128
    buffer: int = 200_000
    warmup: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 200_000
    target_sync: int = 2_000
    updates_per_step: int = 2
    hidden: int = 128
    seed: int = 0
    eval_interval: int = 100


def q_actions(net, obs):
    with torch.no_grad():
        q = net(torch.as_tensor(np.asarray(obs, np.float32)))
    return q.argmax(1).numpy()


def train_dqn(env, cfg: DQNConfig, out_dir, eval_env=None, eval_scenario=None):
    torch.manual_seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        dict(algo="dqn", cfg=asdict(cfg), env=env.config), indent=2), encoding="utf-8")

    n_actions = env.DISCRETE_ACTIONS.shape[0]
    net = QNet(env.obs_dim, n_actions, cfg.hidden)
    target = QNet(env.obs_dim, n_actions, cfg.hidden)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    buf = Replay(cfg.buffer, env.obs_dim, seed=cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 2)
    act_table = env.DISCRETE_ACTIONS

    obs = env.reset_all()
    steps = 0
    grad_steps = 0
    ep_buf = []
    log_f = open(out / "log.jsonl", "w", encoding="utf-8")
    t_start = time.time()
    vec_step = 0

    while steps < cfg.total_steps:
        vec_step += 1
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) *
                  steps / cfg.eps_decay)
        if steps < cfg.warmup or rng.random() < eps:
            a_idx = rng.integers(0, n_actions, env.n)
        else:
            a_idx = q_actions(net, obs)
        obs2, rew, done, trunc, info = env.step(act_table[a_idx])
        buf.add(obs, a_idx, rew, obs2, done.astype(np.float32))
        obs = obs2
        ep_buf.extend(info["episodes"])
        steps += env.n

        loss_val = float("nan")
        if buf.size >= max(cfg.batch, cfg.warmup):
            for _ in range(cfg.updates_per_step):
                o, a, r, o2, d = buf.sample(cfg.batch)
                with torch.no_grad():
                    a_star = net(o2).argmax(1)
                    q_next = target(o2).gather(1, a_star[:, None]).squeeze(1)
                    y = r + cfg.gamma * (1.0 - d) * q_next
                q = net(o).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                loss_val = float(loss.item())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                grad_steps += 1
                if grad_steps % cfg.target_sync == 0:
                    target.load_state_dict(net.state_dict())

        if vec_step % cfg.eval_interval == 0:
            rec = dict(step=steps, vec_step=vec_step, eps=float(eps),
                       ret=float(np.mean([e["ret"] for e in ep_buf])) if ep_buf else float("nan"),
                       success=float(np.mean([e["success"] for e in ep_buf])) if ep_buf else float("nan"),
                       collision=float(np.mean([e["collision"] for e in ep_buf])) if ep_buf else float("nan"),
                       timeout=float(np.mean([e["timeout"] for e in ep_buf])) if ep_buf else float("nan"),
                       q_loss=loss_val,
                       episodes=len(ep_buf), wall=time.time() - t_start)
            ep_buf = []
            if eval_env is not None:
                from .utils import evaluate_policy
                metrics = evaluate_policy(lambda o, e: act_table[q_actions(net, o)], eval_env, eval_scenario)
                rec["eval"] = metrics
                print(f"[dqn {cfg.seed}] step {steps:>8} eps {eps:.2f} succ {rec['success']:.3f} "
                      f"eval_succ {metrics['success']:.3f} ({rec['wall']:.0f}s)", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    torch.save(dict(state_dict=net.state_dict(), cfg=asdict(cfg), obs_dim=env.obs_dim,
                    n_actions=n_actions, hidden=cfg.hidden, algo="dqn"), out / "model.pt")
    log_f.close()
    return net
