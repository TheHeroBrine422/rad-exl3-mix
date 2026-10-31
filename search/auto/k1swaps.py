#!/usr/bin/env python3
"""k1swaps.py K[,K..] GROUP[,GROUP..] -- targeted low-K calibration swaps for kfit2 (evalq2 queue lines on calib-A,
stdout): ours-K on ONE family group over ONE 16-layer band, into sc4.00-e4. Tags use kfit2's cal- form
(cal-sc4.00-e4-ours-KK-GROUP-LbwW), so the fit assigns the damage to that group and band only -- the 2026-10-30 fits
had only mixed-family band swaps (ssm0 = qkv + z + out) and rated early ssm_out / attn_output at K1 as nearly free.
Mix files -> $RXM_WORK/auto/mixes/."""
import re, sys
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")
F = RXM_MODELS; BASE = f"{F}/q38-27b-sc4.00-e4.rad"; G = "golden-calA corpus/calA.jsonl"
GROUPS = {"attn_in": ["attn_qg", "attn_k", "attn_v"], "attn_out": ["attn_output"], "ssm_in": ["ssm_qkv", "ssm_z"],
          "ssm_out": ["ssm_out"], "ffn_gu": ["ffn_gate_up"], "ffn_down": ["ffn_down"]}
SEL = {"attn_in": lambda l: l % 4 == 3, "attn_out": lambda l: l % 4 == 3, "ssm_in": lambda l: l % 4 != 3,
       "ssm_out": lambda l: l % 4 != 3, "ffn_gu": lambda l: True, "ffn_down": lambda l: True}
for K in sys.argv[1].split(","):
    for g in sys.argv[2].split(","):
        for b0 in (0, 16, 32, 48):
            tag = f"cal-sc4.00-e4-ours-K{K}-{g}-L{b0}w16"
            with open(f"{RXM_WORK}/auto/mixes/{tag}.mix", "w") as f:
                for l in range(b0, b0 + 16):
                    if SEL[g](l):
                        for t in GROUPS[g]: f.write(f"blk.{l}.{t}.weight {RXM_OURS}/ours-K{K}.rad\n")
            print(f"{tag} {BASE} {RXM_WORK}/auto/mixes/{tag}.mix {G}")
