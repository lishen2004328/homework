"""Build all paper figures + the numeric digest from results/.

python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("results")
FIG = RES / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "legend.frameon": False,
})

C = {"PPO": "#1f77b4", "DQN": "#d62728", "Reactive": "#ff7f0e",
     "A*": "#2ca02c", "Random": "#7f7f7f"}

GROUPS = [
    ("PPO", ["ppo_s0", "ppo_s1", "ppo_s2"]),
    ("DQN", ["dqn_s0", "dqn_s1", "dqn_s2"]),
    ("PPO-sparse", ["ppo_sparse_s0", "ppo_sparse_s1", "ppo_sparse_s2"]),
    ("PPO-4m", ["ppo_per4_s0", "ppo_per4_s1", "ppo_per4_s2"]),
    ("Reactive", ["reactive"]),
    ("A*", ["astar"]),
    ("Random", ["random"]),
]


def read_curves(runs, key="success"):
    out = {}
    for r in runs:
        log = RES / r / "log.jsonl"
        if not log.exists():
            continue
        steps, vals = [], []
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "eval" in rec:
                steps.append(rec["step"])
                vals.append(rec["eval"][key])
        out[r] = (np.array(steps), np.array(vals))
    return out


def aggregate(curves, grid=None):
    if not curves:
        return None, None, None
    lo = min(c[0][0] for c in curves.values())
    hi = min(c[0][-1] for c in curves.values())
    if grid is None:
        grid = np.linspace(lo, hi, 200)
    interp = [np.interp(grid, c[0], c[1]) for c in curves.values()]
    arr = np.stack(interp)
    return grid, arr.mean(0), arr.std(0)


def panel_curves(ax, group_runs, label, color, key="success"):
    curves = read_curves(group_runs, key)
    grid, mean, std = aggregate(curves)
    if grid is None:
        return
    ax.plot(grid, mean, color=color, lw=1.8, label=label)
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.16, lw=0)


def fig1_learning():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, key, ylab in zip(axes, ["success", "collision"],
                             ["Success rate", "Collision rate"]):
        panel_curves(ax, ["ppo_s0", "ppo_s1", "ppo_s2"], "PPO (ours)", C["PPO"], key)
        panel_curves(ax, ["dqn_s0", "dqn_s1", "dqn_s2"], "Double DQN", C["DQN"], key)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig2_main(summary):
    cond = "main"
    rows = []
    for name, (label, runs) in {
        "PPO": ("PPO (ours)", ["ppo_s0", "ppo_s1", "ppo_s2"]),
        "DQN": ("Double DQN", ["dqn_s0", "dqn_s1", "dqn_s2"]),
        "Reactive": ("Reactive (PF)", ["reactive"]),
        "A*": ("A* oracle", ["astar"]),
        "Random": ("Random", ["random"]),
    }.items():
        sub = summary[(summary.condition == cond) & (summary.run.isin(runs))]
        rows.append((label, sub[["success", "collision", "oob", "spl"]].mean(),
                     sub[["success", "collision", "oob", "spl"]].std()))
    labels = [r[0] for r in rows]
    metrics = ["success", "collision", "oob", "spl"]
    x = np.arange(len(metrics))
    w = 0.16
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for i, (label, mean, std) in enumerate(rows):
        off = (i - 2) * w
        ax.bar(x + off, mean[metrics].values, w, label=label,
               color=list(C.values())[i], alpha=0.92,
               yerr=std[metrics].values if i < 2 else None,
               capsize=2, ecolor="k", error_kw=dict(lw=0.6))
        for xi, v in zip(x + off, mean[metrics].values):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Success ↑", "Collision ↓", "Out of bounds ↓", "SPL ↑"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_main_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig3_ablation(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    panel_curves(ax, ["ppo_s0", "ppo_s1", "ppo_s2"], "PPO (shaped)", C["PPO"])
    panel_curves(ax, ["ppo_sparse_s0", "ppo_sparse_s1", "ppo_sparse_s2"],
                 "PPO (sparse)", C["DQN"])
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval success rate")
    ax.set_title("(a) reward design", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower right")

    ax = axes[1]
    models = {"8 m sensing": ["ppo_s0", "ppo_s1", "ppo_s2"],
              "4 m sensing": ["ppo_per4_s0", "ppo_per4_s1", "ppo_per4_s2"]}
    conds = [("main", "tested @ 8 m"), ("per4", "tested @ 4 m")]
    width = 0.35
    for j, (mname, runs) in enumerate(models.items()):
        vals, errs = [], []
        for c, _ in conds:
            sub = summary[(summary.condition == c) & (summary.run.isin(runs))]
            vals.append(sub.success.mean())
            errs.append(sub.success.std())
        ax.bar(np.arange(2) + (j - 0.5) * width, vals, width,
               yerr=errs, capsize=2, ecolor="k",
               label=f"trained @ {mname}", alpha=0.92,
               color=[C["PPO"], C["Reactive"]][j], error_kw=dict(lw=0.6))
        for k, v in enumerate(vals):
            ax.text(k + (j - 0.5) * width, v + errs[k] + 0.04, f"{v:.2f}",
                    ha="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([c[1] for c in conds])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Success rate")
    ax.set_title("(b) sensing range", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig4_generalisation(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    groups = [("PPO", ["ppo_s0", "ppo_s1", "ppo_s2"], C["PPO"], "o"),
              ("Double DQN", ["dqn_s0", "dqn_s1", "dqn_s2"], C["DQN"], "s"),
              ("Reactive", ["reactive"], C["Reactive"], "^"),
              ("A* oracle", ["astar"], C["A*"], "D")]
    for ax, conds, xt, title in [
            (axes[0], ["d10", "main", "d40"], [10, 25, 40], "(a) obstacle density"),
            (axes[1], ["main", "noise05", "noise15"], [0, 5, 15], "(b) lidar noise σ")]:
        for label, runs, col, mk in groups:
            vals, errs = [], []
            for c in conds:
                sub = summary[(summary.condition == c) & (summary.run.isin(runs))]
                vals.append(sub.success.mean())
                errs.append(sub.success.std() if len(runs) > 1 else 0.0)
            ax.errorbar(xt, vals, yerr=errs, color=col, marker=mk, ms=4, lw=1.4,
                        capsize=2, label=label)
        ax.set_xlabel("Obstacles per map" if "density" in title else "Noise (% of range)")
        ax.set_ylabel("Success rate")
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_generalisation.png", bbox_inches="tight")
    plt.close(fig)


def fig5_trajectories(picks):
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    cols = {"ppo": C["PPO"], "reactive": C["Reactive"], "dqn": C["DQN"], "astar": C["A*"]}
    names = {"ppo": "PPO", "reactive": "人工势场", "dqn": "Double DQN", "astar": "A*（真值地图）"}
    fig, axes = plt.subplots(1, len(picks), figsize=(2.35 * len(picks), 2.4))
    axes = np.atleast_1d(axes).ravel()
    handles = {}
    for ax, tag, (scen_idx, kinds) in zip(axes, "ABCD", picks):
        data = {k: np.load(RES / f"traj_{k}.npz", allow_pickle=True) for k in kinds}
        ref = data[kinds[0]]
        h = float(ref["map_half"])
        ax.add_patch(plt.Rectangle((-h, -h), 2 * h, 2 * h, fill=False, lw=0.8, ec="0.4"))
        for c, r in zip(ref["obs_c"][scen_idx], ref["obs_r"][scen_idx]):
            ax.add_patch(plt.Circle(c, r, color="0.55", alpha=0.85, lw=0))
        ax.plot(*ref["start"][scen_idx, :2], "o", mfc="k", mec="k", ms=4, zorder=5)
        ax.plot(*ref["goal"][scen_idx, :2], "*", mfc="#2ca02c", mec="#2ca02c", ms=9, zorder=5)
        for k in kinds:
            d = data[k]
            f = d["frames"][:, scen_idx, :2]
            ok = bool(d["outcomes"][scen_idx])
            ln, = ax.plot(f[:, 0], f[:, 1], color=cols[k], lw=1.5,
                          ls="-" if ok else "--", zorder=4)
            handles.setdefault(k, ln)
        dist = float(np.linalg.norm(ref["goal"][scen_idx, :2] - ref["start"][scen_idx, :2]))
        fail = [names[k] for k in kinds if not bool(data[k]["outcomes"][scen_idx])]
        title = f"场景 {tag}　{dist:.1f} m"
        if fail:
            title += "　（" + "、".join(fail) + "失败）"
        ax.set_title(title, fontsize=6.4, fontproperties=fp)
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.legend(list(handles.values()), [names[k] for k in handles], loc="lower center",
               ncol=4, frameon=False, fontsize=7.0, prop=fp)
    fig.savefig(FIG / "fig5_trajectories.png", bbox_inches="tight")
    plt.close(fig)



def fig0_method():
    """Schematic of the closed loop (Chinese labels via an explicit font file)."""
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.35, 3.0)
    ax.axis("off")
    boxes = [(0.15, "观测 $o_t$\n48 维：激光 32 + 状态 16"),
             (2.75, "策略网络 $\\pi_\\theta$\nActor-Critic 128×128"),
             (5.35, "加速度指令 $a_t$\n3 维，$[-1,1]^3$"),
             (7.95, "无人机动力学\n双积分，dt = 0.1 s")]
    w, h, y = 2.2, 0.85, 1.55
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#eef3fa", ec="0.35", lw=0.9))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=6.8, linespacing=1.45, fontproperties=fp)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w
        ax.annotate("", xy=(boxes[i + 1][0], y + h / 2), xytext=(x0, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.25"))
    ax.annotate("", xy=(boxes[0][0] + w / 2, y), xytext=(boxes[3][0] + w / 2, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#1f77b4",
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(5.2, 0.55, "奖励 $r_t$、下一观测 $o_{t+1}$", color="#1f77b4", fontsize=7.2,
            ha="center", fontproperties=fp)
    ax.text(0.15, 2.72, "奖励：势函数塑形 + 终端奖励 + 近距惩罚（碰撞/越界 -3，到达 +3）",
            fontsize=7.2, color="0.25", fontproperties=fp)
    fig.savefig(FIG / "fig0_method.png", bbox_inches="tight")
    plt.close(fig)


def digest(summary):
    num = {}
    for cond in sorted(summary.condition.unique()):
        num[cond] = {}
        for label, runs in [("PPO", ["ppo_s0", "ppo_s1", "ppo_s2"]),
                            ("DQN", ["dqn_s0", "dqn_s1", "dqn_s2"]),
                            ("PPO-sparse", ["ppo_sparse_s0", "ppo_sparse_s1", "ppo_sparse_s2"]),
                            ("PPO-4m", ["ppo_per4_s0", "ppo_per4_s1", "ppo_per4_s2"]),
                            ("Reactive", ["reactive"]), ("A*", ["astar"]),
                            ("Random", ["random"])]:
            sub = summary[(summary.condition == cond) & (summary.run.isin(runs))]
            if len(sub):
                num[cond][label] = {m: [float(sub[m].mean()), float(sub[m].std())]
                                    for m in ["success", "collision", "oob", "timeout",
                                              "spl", "steps", "path_ratio", "energy",
                                              "clearance_p05"]}
    (RES / "numbers.json").write_text(json.dumps(num, indent=2), encoding="utf-8")
    return num


if __name__ == "__main__":
    summary = pd.read_csv(RES / "summary.csv")
    num = digest(summary)
    fig0_method()
    fig1_learning()
    fig2_main(summary)
    fig3_ablation(summary)
    fig4_generalisation(summary)
    picks = [(3, ["ppo", "reactive", "astar"]),
             (2, ["ppo", "reactive", "astar"]),

             (7, ["ppo", "dqn", "astar"])]
    fig5_trajectories(picks)
    print("figures ->", FIG)
    print(json.dumps({c: {lab: {m: round(md[m][0], 3)
                               for m in ("success", "collision", "spl")}
                        for lab, md in vv.items()}
                      for c, vv in num.items()}, indent=1))
