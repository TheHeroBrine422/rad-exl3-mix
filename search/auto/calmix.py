#!/usr/bin/env python3
"""calmix.py ours K | emb -- calibration jobs for the overnight pipeline, as evalq2 queue lines on calib-A (stdout).
  ours K: the whole ours-K container (a-ref-ours-KK: smoke test + whole-model point) and three band swaps of it into
          sc4.00-e4 (a-xp-oursKK-{attn48,ssm0,ffn32}: late full attention, early GDN, mid MLP) -> the ours family
          factors (kfit2: ours_lo K1-3, ours_mid K4-6, ours_hi K7-8)
  emb:    sc4.00-e4 with each new embedding: Lloyd 1/2/7/8 bit (sc4.00-eB donors) and bf16 (the bf16 container)"""
import re, sys, json, glob
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")
F, KN = RXM_MODELS, f"{RXM_WORK}/knap/mixes"
BASE = f"{F}/q38-27b-sc4.00-e4.rad"
G = "golden-calA corpus/calA.jsonl"
W = sorted({json.loads(l)["w"] for p in glob.glob(f"{RXM_WORK}/knap/hproxy-c*.jsonl") for l in open(p)})
if sys.argv[1] == "ours":
    K = int(sys.argv[2]); src = f"{RXM_OURS}/ours-K{K}.rad"
    print(f"a-ref-ours-K{K} {src} - {G}")
    bands = {"attn48": r"blk\.(4[89]|5\d|6[0-3])\.attn_(qg|k|v|output)\.weight$",
             "ssm0": r"blk\.([0-9]|1[0-5])\.ssm_(qkv|z|out)\.weight$",
             "ffn32": r"blk\.(3[2-9]|4[0-7])\.ffn_(gate_up|down)\.weight$"}
    for b, rx in bands.items():
        n = f"xp-oursK{K}-{b}"
        with open(f"{KN}/{n}.mix", "w") as f:
            for w in W:
                if re.match(rx, w): f.write(f"{w} {src}\n")
        print(f"a-{n} {BASE} {KN}/{n}.mix {G}")
elif sys.argv[1] == "emb":
    for b in (1, 2, 7, 8, "bf16"):
        d = f"{F}/q38-27b-sc4.00-e{b}.rad" if b != "bf16" else f"{F}/q38-27b-bf16.rad"
        t = f"emb-sc4.00-e4-sc4.00-e{b}" if b != "bf16" else "emb-sc4.00-e4-bf16"
        with open(f"{RXM_WORK}/auto/mixes/{t}.mix", "w") as f: f.write(f"token_embd.weight {d}\n")
        print(f"{t} {BASE} {RXM_WORK}/auto/mixes/{t}.mix {G}")
