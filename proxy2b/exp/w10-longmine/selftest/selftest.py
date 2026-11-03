#!/usr/bin/env python3
"""CPU self-test of longmine.py on synthetic long traces (2 docs x 250k positions, orig 905 / 2675, 50% gen) and the
REAL V2 container byte table (data/v2/radls.jsonl): 14 mixes drawing attention / GDN / FFN tensors from different
plain containers (2.00-6.00), plus one arm scored on a 100k-capped golden (row alignment by (orig, position)).
Arm KL per row = shared base x 2^-(trunk bits - 4) x exp(beta_attn (attn - trunk bits) [pos >= 32k]) x row noise.
Case 1 beta_attn = 0.3 -> attn flagged near x1.35, gdn not; case 2 beta = 0 -> nothing flagged."""
import json, math, os, subprocess, sys, tempfile
import numpy as np
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADLS = os.path.join(X, "../../../data/v2/radls.jsonl")
ok = True
def check(c, msg):
    global ok; ok &= bool(c); print(("PASS " if c else "FAIL ") + msg)
RL = {json.loads(l)["c"]: json.loads(l)["w"] for l in open(RADLS)}
bf = next(v for c, v in RL.items() if c.endswith("bf16.rad")); PAR = {w: b // 2 for w, b in bf.items()}
PL = [c for c in RL if "plain" in c]; P4 = next(c for c in PL if "plain4.00" in c)
FAM = {"attn_qg": "a", "attn_k": "a", "attn_v": "a", "attn_output": "a", "ssm_qkv": "g", "ssm_z": "g", "ssm_out": "g",
       "ffn_gate_up": "f", "ffn_down": "f"}

def golden(d, cap):
    os.makedirs(d); docs, r = [], 0
    for o in (905, 2675):
        nt = min(250001, cap); docs.append({"source": "agent", "tokens": nt, "score_from": 0, "row0": r, "orig": o}); r += nt - 1
    json.dump({"rows": r, "docs": docs}, open(f"{d}/kld.json", "w"))
    return r

def run_case(beta, T):
    rng = np.random.default_rng(7)
    R = golden(f"{T}/g", 1 << 30); Rc = golden(f"{T}/gc", 100001)
    gen = (rng.random(R) < 0.5).astype(np.uint8); gen.tofile(f"{T}/g/gen.u8")
    pos = np.concatenate([np.arange(250000)] * 2)
    capsel = np.concatenate([np.arange(100000), 250000 + np.arange(100000)])
    gen[capsel].tofile(f"{T}/gc/gen.u8")
    base = rng.gamma(1.5, 0.01, R) * (1 + pos / 1e5)
    noise = lambda: rng.lognormal(-0.125, 0.5, R)
    lines = []
    def write(tag, kl, gd, src, sel=None):
        v = kl if sel is None else kl[sel]
        np.stack([v, np.ones(len(v)), np.ones(len(v)), np.zeros(len(v))], 1).astype(np.float32).tofile(f"{T}/{tag}.f32")
        lines.append(f"{tag}\t{T}/{tag}.f32\t{gd}\t{src}")
    write("ref-plain4", base * noise(), f"{T}/g", P4)
    for i in range(15):
        ch = {k: PL[rng.integers(len(PL))] for k in "agf"}
        mix = f"{T}/m{i}.mix"
        with open(mix, "w") as f:
            for w in PAR:
                p = w.split(".")
                c = ch[FAM[p[2]]] if len(p) == 4 and p[0] == "blk" and p[2] in FAM else P4
                f.write(f"{w} {c}\n")
        bits = {}
        for k in "agf":
            ws = [w for w in PAR if len(w.split(".")) == 4 and w.split(".")[2] in FAM and FAM[w.split(".")[2]] == k]
            bits[k] = (sum(RL[ch[k]][w] * 8 for w in ws) / sum(PAR[w] for w in ws), sum(PAR[w] for w in ws))
        trunk = sum(b * n for b, n in bits.values()) / sum(n for _, n in bits.values())
        kl = base * 2.0 ** -(trunk - 4) * np.exp(beta * (bits["a"][0] - trunk) * (pos >= 32768)) * noise()
        if i < 14: write(f"mix{i}", kl, f"{T}/g", mix)
        else: write("mix14-capped", kl, f"{T}/gc", mix, capsel)
    open(f"{T}/manifest.tsv", "w").write("# synthetic\n" + "\n".join(lines) + "\n")
    r = subprocess.run([sys.executable, f"{X}/longmine.py", f"{T}/manifest.tsv", RADLS, f"{T}/out.json", "--boot", "300"],
                       capture_output=True, text=True)
    print(r.stdout[-2500:]); print(r.stderr[-2000:])
    return json.load(open(f"{T}/out.json"))

o = run_case(0.3, tempfile.mkdtemp())
g = o["regression"]
check(g["attn_minus_trunk"]["flag"] and abs(g["attn_minus_trunk"]["x_per_bit"] - math.exp(0.3)) < 0.1,
      f"beta 0.3: attn flagged, x{g['attn_minus_trunk']['x_per_bit']:.3f} (planted x{math.exp(0.3):.3f})")
check(not g["gdn_minus_trunk"]["flag"], f"beta 0.3: gdn not flagged (x{g['gdn_minus_trunk']['x_per_bit']:.3f})")
cap = o["arms"]["mix14-capped"]["bands"]
check("128k+" not in cap and "64-128k" in cap and "c" in o["arms"]["mix14-capped"], "capped arm aligned: bands to 64-128k only, c computed")
o = run_case(0.0, tempfile.mkdtemp())
g = o["regression"]
check(not g["attn_minus_trunk"]["flag"] and not g["gdn_minus_trunk"]["flag"], "beta 0: nothing flagged")
# mkmanifest: golden by row count, mix before container, unresolved listed
T = tempfile.mkdtemp(); R = golden(f"{T}/golden-long2", 1 << 30); golden(f"{T}/golden-long2-c100000", 100001)
for d in ("sc", "mx"): os.makedirs(f"{T}/{d}")
for tag, n in (("td-plain4.00-e4", R), ("longcap-knap-s7-sc4.00-e4-2.50", 2 * 100000), ("v2-3.00", R), ("odd", 12345)):
    np.zeros((n, 4), np.float32).tofile(f"{T}/sc/{tag}.f32")
open(f"{T}/mx/knap-s7-sc4.00-e4-2.50.mix", "w").write("output.weight " + P4 + "\n")
r = subprocess.run([sys.executable, f"{X}/mkmanifest.py", RADLS, f"{T}/sc", f"{T}/golden-long2*", f"{T}/mx"], capture_output=True, text=True)
print(r.stdout)
L = {l.split("\t")[0]: l.split("\t") for l in r.stdout.splitlines() if not l.startswith("#")}
check(L.get("td-plain4.00-e4", [""] * 4)[3] == P4 and L["td-plain4.00-e4"][2].endswith("golden-long2"), "uniform container by tag end, full golden by rows")
check(L.get("longcap-knap-s7-sc4.00-e4-2.50", [""] * 4)[3].endswith(".mix") and L["longcap-knap-s7-sc4.00-e4-2.50"][2].endswith("c100000"),
      "V2 mix wins over the sc4.00-e4 substring; capped golden by rows")
check("# UNRESOLVED v2-3.00" in r.stdout and "# UNRESOLVED odd" in r.stdout, "unresolved arms listed, not guessed")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
