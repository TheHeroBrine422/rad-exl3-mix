#!/usr/bin/env python3
"""mkoff.py OSR.jsonl [THRESH] -- ourq --out-scales-off list: the gate / up projections whose upstream output-scale
ratio (osr.py, normalized inputs) exceeds THRESH (default 1.05, exllamav3's out_scales_max_ratio)."""
import json, sys
th = float(sys.argv[2]) if len(sys.argv) > 2 else 1.05
for r in sorted((json.loads(l) for l in open(sys.argv[1])), key=lambda r: r["layer"]):
    for p in ("gate", "up"):
        if r[f"{p}_norm"] > th:
            print(f"model.language_model.layers.{r['layer']}.mlp.{p}_proj")
