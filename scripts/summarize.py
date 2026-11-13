"""Collapse per-run evaluation json files into a single CSV.

python scripts/summarize.py --root results --out results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = ["success", "collision", "oob", "timeout", "spl", "steps", "path",
          "path_ratio", "energy", "clearance_p05", "episodes"]


def run_meta(d: Path):
    cfg_file = d / "config.json"
    meta = dict(run=d.name, algo=d.name.split("_")[0], seed="", reward="", perception="")
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        meta["algo"] = cfg["algo"]
        meta["seed"] = cfg["cfg"].get("seed", "")
        meta["reward"] = cfg["env"].get("reward_mode", "")
        meta["perception"] = cfg["env"].get("perception_range", "")
    else:
        # baseline runs are named after their policy
        meta["algo"] = d.name
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.glob("eval_*.json")):
            # one json per run: {condition: {metric: value, ...}}
            res = json.loads(f.read_text(encoding="utf-8"))
            for condition, metrics in res.items():
                if not isinstance(metrics, dict) or "success" not in metrics:
                    continue
                row = dict(run_meta(d))
                row["condition"] = condition
                for k in FIELDS:
                    row[k] = metrics.get(k, "")
                rows.append(row)
    if not rows:
        raise SystemExit("no eval_*.json found under results/ -- run scripts/evaluate.py first")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
