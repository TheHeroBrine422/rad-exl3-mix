#!/usr/bin/env python3
"""rxqdrift107.py RXQ-ARGS... -- #141-02 (../w107.md): `python -m rxq ...` with exllamav3's LDLQ_DRIFT table points set to
the proposed sub-1 drift table (K0.125 1.348, K0.5 1.184, K1 1.089; W107_DRIFT = json {K: d} overrides) before rxq's
engine builds its log-linear table from it. Nothing else changes (the settings stamp in the tensors' metadata does not
see this override: the arm is named by its output dir)."""
import json, os, sys
import exllamav3.modules.quant.exl3_lib.quantize as Q

T = {0.125: 1.348, 0.5: 1.184, 1.0: 1.089}
if os.environ.get("W107_DRIFT"): T = {float(k): float(v) for k, v in json.loads(os.environ["W107_DRIFT"]).items()}
Q.LDLQ_DRIFT.pop(1, None); Q.LDLQ_DRIFT.update(T)
print(f"rxqdrift107: LDLQ_DRIFT = {dict(sorted(Q.LDLQ_DRIFT.items()))}", flush=True)
from rxq.__main__ import main  # noqa: E402
sys.exit(main(sys.argv[1:]))
