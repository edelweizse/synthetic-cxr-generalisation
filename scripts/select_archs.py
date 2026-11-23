"""Pick the architectures carried into the data experiments, by the pre-registered rule in
cxr.analysis.select_archs (best CNN + best transformer by mean E1 A-validation loss).

    python scripts/select_archs.py   -> configs/selected_archs.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxr import PROJECT_ROOT
from cxr.analysis import collect_runs, select_archs

runs = collect_runs()
sel = select_archs(runs, k=2)
order = runs[runs.experiment == "E1"].drop_duplicates(["arch", "seed"]).groupby("arch").best_val_loss.mean().sort_values()
out = {"selected": sel, "rule": "best CNN + best transformer by mean E1 A-validation BCE",
       "e1_val_loss_by_arch": order.round(5).to_dict()}
(PROJECT_ROOT / "configs" / "selected_archs.json").write_text(json.dumps(out, indent=2))
print(" ".join(sel))
