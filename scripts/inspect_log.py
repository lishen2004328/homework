"""Print the tail of a training log / evaluation json (debug helper)."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if p.suffix == ".json":
    print(json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=1, ensure_ascii=False))
else:
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ev = [l for l in lines if "eval" in l]
    print("updates:", len(lines))
    for l in ev[-3:]:
        print(f"\nstep {l['step']}  ret {l.get('ret', float('nan')):.2f}  "
              f"train_succ {l.get('success', float('nan')):.3f} "
              f"coll {l.get('collision', float('nan')):.3f} "
              f"timeout {l.get('timeout', float('nan')):.3f} "
              f"ent {l.get('entropy', float('nan')):.3f} "
              f"log_std {l.get('log_std', float('nan')):.2f} "
              f"vloss {l.get('v_loss', float('nan')):.2f}")
        print(json.dumps(l["eval"], ensure_ascii=False))
