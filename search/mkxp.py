#!/usr/bin/env python3
"""mkxp.py DIR -- the model-form experiments as mix files DIR/mixes/xp-*.mix (base sc4.00-e4; whole-container
backgrounds are written out weight by weight), plus queue lines "a-xp-NAME BASE.rad MIX" on stdout.

  background swaps: the same band moved one step down on sc5.00 / sc3.00 / sc2.00 backgrounds
                    -> constant dKLD = additive, constant ratio = multiplicative (kfit's g / depth terms)
  K ladder:         one band of the base taken from sc5.00 / sc2.00 / sc1.40 (sc3.00 exists as cal-*-w16)
                    -> dKLD vs proxy error, per-tensor convexity (kfit's q)"""
import json, re, sys, glob
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
D = sys.argv[1]
F = RXM_MODELS
rad = lambda c: f"{F}/q38-27b-{c}.rad"
W = sorted({json.loads(l)["w"] for p in glob.glob(f"{D}/hproxy*.jsonl") for l in open(p)})
BANDS = {"ffngu32": lambda w: re.match(r"blk\.(3[2-9]|4[0-7])\.ffn_gate_up\.weight$", w),
         "down0": lambda w: re.match(r"blk\.([0-9]|1[0-5])\.ffn_down\.weight$", w)}
def write(name, assign, emb):
    with open(f"{D}/mixes/xp-{name}.mix", "w") as f:
        for w in W:
            c = assign(w)
            if c != "sc4.00-e4": f.write(f"{w} {rad(c)}\n")
        if emb != "sc4.00-e4": f.write(f"token_embd.weight {rad(emb)}\n")
    print(f"a-xp-{name} {rad('sc4.00-e4')} {D}/mixes/xp-{name}.mix")
STEP = {"sc5.00-e5": "sc4.00-e4", "sc3.00-e4": "sc2.00-e3", "sc2.00-e3": "sc1.40-e3"}
for bg, down in STEP.items():
    for b, sel in BANDS.items():
        write(f"{bg}-{b}", lambda w, bg=bg, down=down, sel=sel: down if sel(w) else bg, bg)
for donor in ("sc5.00-e5", "sc2.00-e3", "sc1.40-e3"):
    write(f"ladder-ffngu32-{donor}", lambda w, donor=donor: donor if BANDS["ffngu32"](w) else "sc4.00-e4", "sc4.00-e4")
