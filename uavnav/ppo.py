"""PPO (Schulman et al., 2017) with GAE for the vectorised UAV task."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, init_log_std=-0.7):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        for lin in (self.trunk[0], self.trunk[2]):
            _ortho(lin, np.sqrt(2.0))
        _ortho(self.pi, 0.01)
        _ortho(self.v, 1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def distribution(self, x):
        mean, value = self.forward(x)
        std = torch.exp(self.log_std.clamp(-3.0, 0.0))
        return Normal(mean, std), value


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    rollout_len: int = 128
    epochs: int = 6
    minibatch: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 128
    init_log_std: float = -0.7
    seed: int = 0
    log_every: int = 1


def train_ppo(env, cfg: PPOConfig, out_dir, eval_env=None, eval_scenario=None,
              eval_every_updates=10):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        dict(algo="ppo", cfg=asdict(cfg), env=env.config), indent=2), encoding="utf-8")

    net = ActorCritic(env.obs_dim, env.act_dim, cfg.hidden, cfg.init_log_std)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    n, T = env.n, cfg.rollout_len
    obs = env.reset_all()
    obs_t = torch.as_tensor(obs)
    steps = 0
    update = 0
    ep_buf = []
    log_f = open(out / "log.jsonl", "w", encoding="utf-8")
    t_start = time.time()

    while steps < cfg.total_steps:
        update += 1
        b_obs = torch.zeros(T, n, env.obs_dim)
        b_act = torch.zeros(T, n, env.act_dim)
        b_logp = torch.zeros(T, n)
        b_val = torch.zeros(T, n)
        b_rew = torch.zeros(T, n)
        b_done = torch.zeros(T, n)
        b_vnext = torch.zeros(T, n)

        for t in range(T):
            with torch.no_grad():
                dist, value = net.distribution(obs_t)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
            obs_next, rew, done, trunc, info = env.step(act.numpy())
            ep_buf.extend(info["episodes"])
            with torch.no_grad():
                v_next = net(torch.as_tensor(obs_next))[1]
                if len(info["trunc"]):
                    idx = info["trunc"]
                    v_next[idx] = net(torch.as_tensor(info["obs_pre"][idx]))[1]
            b_obs[t] = obs_t
            b_act[t] = act
            b_logp[t] = logp
            b_val[t] = value
            b_rew[t] = torch.as_tensor(rew)
            b_done[t] = torch.as_tensor(done.astype(np.float32))
            b_vnext[t] = v_next
            obs_t = torch.as_tensor(obs_next)
            steps += n

        # ---- GAE
        adv = torch.zeros(T, n)
        last = torch.zeros(n)
        for t in reversed(range(T)):
            delta = b_rew[t] + cfg.gamma * b_vnext[t] * (1.0 - b_done[t]) - b_val[t]
            last = delta + cfg.gamma * cfg.gae_lambda * (1.0 - b_done[t]) * last
            adv[t] = last
        ret = adv + b_val

        # ---- learning rate annealing
        frac = 1.0 - steps / cfg.total_steps
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (0.1 + 0.9 * frac)

        obs_flat = b_obs.reshape(T * n, -1)
        act_flat = b_act.reshape(T * n, -1)
        logp_flat = b_logp.reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        pi_losses, v_losses, ents, kls = [], [], [], []
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * n)
            for s in range(0, T * n, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                dist, val = net.distribution(obs_flat[mb])
                logp = dist.log_prob(act_flat[mb]).sum(-1)
                ratio = torch.exp(logp - logp_flat[mb])
                s1 = ratio * adv_flat[mb]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_flat[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * ((val - ret_flat[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                ents.append(ent.item())
                kls.append(float((((ratio - 1) - (logp - logp_flat[mb])).mean()).detach()))

        rec = dict(step=steps, update=update,
                   ret=float(np.mean([e["ret"] for e in ep_buf])) if ep_buf else float("nan"),
                   success=float(np.mean([e["success"] for e in ep_buf])) if ep_buf else float("nan"),
                   collision=float(np.mean([e["collision"] for e in ep_buf])) if ep_buf else float("nan"),
                   timeout=float(np.mean([e["timeout"] for e in ep_buf])) if ep_buf else float("nan"),
                   episodes=len(ep_buf), pi_loss=float(np.mean(pi_losses)),
                   v_loss=float(np.mean(v_losses)), entropy=float(np.mean(ents)),
                   approx_kl=float(np.mean(kls)),
                   log_std=float(net.log_std.mean().item()),
                   wall=time.time() - t_start)
        ep_buf = []

        if eval_env is not None and (update % eval_every_updates == 0):
            from .utils import evaluate_policy
            net.eval()
            metrics = evaluate_policy(lambda o, e: policy_actions(net, o), eval_env, eval_scenario)
            net.train()
            rec["eval"] = metrics
            print(f"[ppo {cfg.seed}] step {steps:>8} ret {rec['ret']:.2f} succ {rec['success']:.3f} "
                  f"eval_succ {metrics['success']:.3f} spl {metrics['spl']:.3f} "
                  f"({rec['wall']:.0f}s)", flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    torch.save(dict(state_dict=net.state_dict(), cfg=asdict(cfg),
                    obs_dim=env.obs_dim, act_dim=env.act_dim, hidden=cfg.hidden),
               out / "model.pt")
    log_f.close()
    return net


def policy_actions(net, obs, deterministic=True):
    """Action used at evaluation time (mean of the Gaussian policy)."""
    with torch.no_grad():
        dist, _ = net.distribution(torch.as_tensor(np.asarray(obs, np.float32)))
        mean = dist.mean
    return np.clip(mean.numpy(), -1.0, 1.0)
