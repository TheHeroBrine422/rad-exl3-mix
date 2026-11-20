#!/usr/bin/env python3
"""asmcount.py ASMDIR CLANG SRC [MM M A4] -- instructions of each candidate's GEMV inner loop (device asm, gfx1201).
Compiles SRC once a candidate (-S, device only), finds gemv<Cand, MM, M, A4>, takes the innermost backward-branch loop
that holds the dot (the column loop: one step = 16 weights a lane) and counts its instructions by class.
Prints JSON lines {cand, valu, load_global, load_lds, scalar, per_weight (valu / 16)}."""
import json, os, re, subprocess, sys
d, clang, src = sys.argv[1:4]
MM, M, A4 = (sys.argv[4:7] + ["32", "1", "0"])[:3] if len(sys.argv) >= 7 else ("32", "1", "0")
C = "Raw Base Mad MadF Mad4 Mad4F Mul1I8 Int4Grid Qtip1Mad Qtip3Inst Cptc6Lds Cptc6Glb Cptc6Eq3 Cptc6FLds Cptc6FGlb Cptc5Lds Cptc5Glb Cptc5Reg St6Lds St6Glb St8Lds St8Glb St10Lds St10Glb St12Lds St12Glb CptcMin Lc1Lds Lc1Glb".split()
os.makedirs(d, exist_ok=True)
for c in C:
    s = f"{d}/{c}.s"
    if not os.path.exists(s):
        subprocess.run([clang, "-x", "hip", "--cuda-device-only", "-S", "--offload-arch=gfx1201", "-O3", "-std=c++17", f"-DCAND={c}", src, "-o", s], check=True, stderr=subprocess.DEVNULL)
    L = open(s).read().split("\n")
    pat = re.compile(rf"^_Z4gemvI.*Li{MM}ELi{M}ELi{A4}EEv.*:")
    st = [i for i, l in enumerate(L) if pat.match(l)]
    if not st:
        continue
    a = st[0]; b = next(i for i in range(a + 1, len(L)) if L[i].startswith(".Lfunc_end"))
    F = L[a:b]
    lab = {m.group(1): i for i, l in enumerate(F) if (m := re.match(r"^(\.LBB\w+):", l))}
    best = None
    for i, l in enumerate(F):
        m = re.search(r"\b(s_cbranch\w+|s_branch)\s+(\.LBB\w+)", l)
        if m and m.group(2) in lab and lab[m.group(2)] < i:
            body = [x.split(";")[0].strip() for x in F[lab[m.group(2)]:i + 1]]
            body = [x for x in body if x and not x.endswith(":") and not x.startswith(".")]
            if any("v_dot" in x for x in body) and (best is None or len(body) < len(best)):
                best = body
    if best is None:
        continue
    op = [x.split()[0] for x in best]
    r = {"cand": c, "MM": int(MM), "M": int(M), "A4": int(A4),
         "valu": sum(o.startswith("v_") for o in op), "load_global": sum(o.startswith("global_load") for o in op),
         "load_lds": sum(o.startswith("ds_") for o in op), "scalar": sum(o.startswith("s_") for o in op), "total": len(op)}
    r["valu_per_weight"] = round(r["valu"] / 16, 2)
    print(json.dumps(r), flush=True)
