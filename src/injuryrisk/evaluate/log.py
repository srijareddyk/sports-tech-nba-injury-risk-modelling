"""Append-only log of every configuration fit (spec 6.3).

The fantasy project scored 73 configurations against one benchmark, and its
"winners" were indistinguishable from best-of-73 luck. Every fit, on validation
or test, lands in ``experiments/log.csv`` so the number of tries is always known.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from injuryrisk.paths import EXPERIMENTS

LOG = EXPERIMENTS / "log.csv"
FIELDS = ["timestamp", "config_hash", "stage", "target", "horizon", "test_season", "eval_on",
          "model", "params", "n", "positives", "pr_auc", "brier", "roc_auc"]


def log_fit(**row) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    row = {k: row.get(k, "") for k in FIELDS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(row["params"], dict):
        row["params"] = json.dumps(row["params"], sort_keys=True)
    for k in ("pr_auc", "brier", "roc_auc"):
        if isinstance(row[k], float):
            row[k] = f"{row[k]:.6f}"
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def config_count() -> dict:
    """How many fits, and how many distinct (model, params) configs, so far."""
    if not LOG.exists():
        return {"fits": 0, "distinct_configs": 0}
    with open(LOG, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {"fits": len(rows), "distinct_configs": len({(r["model"], r["params"]) for r in rows})}
