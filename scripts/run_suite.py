"""Run the full experiment matrix (training + evaluation + figures).

python scripts/run_suite.py [--jobs 6] [--skip-train]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MAIN_STEPS = 2_000_000
ABL_STEPS = 1_000_000
SEEDS = [0, 1, 2]

TRAIN = []
for s in SEEDS:
    TRAIN.append(["scripts/train.py", "--algo", "ppo", "--seed", str(s),
                  "--steps", str(MAIN_STEPS), "--out", f"results/ppo_s{s}",
                  "--eval-every", "10", "--eval-episodes", "128"])
    TRAIN.append(["scripts/train.py", "--algo", "dqn", "--seed", str(s),
                  "--steps", str(MAIN_STEPS), "--out", f"results/dqn_s{s}",
                  "--eval-every", "100", "--eval-episodes", "128"])
    TRAIN.append(["scripts/train.py", "--algo", "ppo", "--seed", str(s),
                  "--steps", str(ABL_STEPS), "--reward", "sparse",
                  "--out", f"results/ppo_sparse_s{s}", "--eval-every", "10",
                  "--eval-episodes", "128"])
    TRAIN.append(["scripts/train.py", "--algo", "ppo", "--seed", str(s),
                  "--steps", str(ABL_STEPS), "--perception", "4.0",
                  "--out", f"results/ppo_per4_s{s}", "--eval-every", "10",
                  "--eval-episodes", "128"])

RUNS = ([f"ppo_s{s}" for s in SEEDS] + [f"dqn_s{s}" for s in SEEDS] +
        [f"ppo_sparse_s{s}" for s in SEEDS] + [f"ppo_per4_s{s}" for s in SEEDS])

EVAL = ([("ppo", None, f"results/ppo_s{s}") for s in SEEDS] +
        [("dqn", None, f"results/dqn_s{s}") for s in SEEDS] +
        [("ppo", None, f"results/ppo_sparse_s{s}") for s in SEEDS] +
        [("ppo", None, f"results/ppo_per4_s{s}") for s in SEEDS] +
        [("reactive", None, "results/reactive"), ("astar", None, "results/astar"),
         ("random", None, "results/random")])

CONDITIONS = "main,d10,d40,scale,noise05,noise15,per4"


def run(cmd, jobs, logdir):
    logdir.mkdir(parents=True, exist_ok=True)
    name = cmd[cmd.index("--out") + 1].replace("/", "_") if "--out" in cmd else "x"
    log = open(logdir / f"{name}.log", "w", encoding="utf-8")
    return subprocess.Popen([PY, "-X", "utf8"] + cmd, cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT)


def wait_batch(procs, jobs):
    while procs:
        running = [p for p in procs if p.poll() is None]
        if len(running) < jobs:
            # launch next jobs
            pass
        if not running:
            break
        time.sleep(3)
    return [p.returncode for p in procs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--only-eval", action="store_true")
    args = ap.parse_args()

    logs = ROOT / "results" / "_logs"
    if not args.only_eval:
        queue = list(TRAIN)
        procs, failed = [], []
        t0 = time.time()
        while queue or procs:
            while queue and len(procs) < args.jobs:
                cmd = queue.pop(0)
                print("launch:", " ".join(cmd[1:6]), flush=True)
                procs.append((cmd, run(cmd, args.jobs, logs)))
            time.sleep(5)
            for i in range(len(procs) - 1, -1, -1):
                cmd, p = procs[i]
                if p.poll() is not None:
                    if p.returncode != 0:
                        failed.append(" ".join(cmd))
                        print("FAILED:", " ".join(cmd), flush=True)
                    procs.pop(i)
        print(f"training finished in {time.time() - t0:.0f}s, failures={len(failed)}")

    # ---- evaluation (same bounded-parallelism pool as training)
    t0 = time.time()
    cmds = []
    for kind, _, out in EVAL:
        c = ["scripts/evaluate.py", "--kind", kind, "--out", out,
             "--conditions", CONDITIONS, "--episodes", "256"]
        if kind in ("ppo", "dqn"):
            c += ["--model", f"{out}/model.pt"]
        cmds.append(c)

    procs, failed = [], []
    while cmds or procs:
        while cmds and len(procs) < args.jobs:
            cmd = cmds.pop(0)
            print("eval:", " ".join(cmd[2:6]), flush=True)
            procs.append((cmd, run(cmd, args.jobs, logs)))
        time.sleep(2)
        for i in range(len(procs) - 1, -1, -1):
            cmd, p = procs[i]
            if p.poll() is not None:
                if p.returncode != 0:
                    failed.append(" ".join(cmd))
                    print("EVAL FAILED:", " ".join(cmd), flush=True)
                procs.pop(i)
    print(f"evaluation finished in {time.time() - t0:.0f}s, failures={len(failed)}", flush=True)

    subprocess.run([PY, "-X", "utf8", "scripts/summarize.py", "--root", "results",
                    "--out", "results/summary.csv"], cwd=ROOT, check=False)
    if failed:
        print("FAILED RUNS:\n" + "\n".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
