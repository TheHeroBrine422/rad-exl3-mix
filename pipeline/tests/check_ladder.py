#!/usr/bin/env python3
"""check_ladder.py [RQAH_CHECKOUT] -- rxq/ladder.py, the ladder's job generation (CPU, seconds):
  levels   50 levels from the settings file (int8-only 11-09: mul1 out, i8 K7.5 / K8 in): pass 1 20 (i8 quarter steps
           K1.25-5, half steps K5.5-7), pass 2 23, no i4 twins (A4 out, w93 11-08), pass 3 7 (sub-1, i8 since #126;
           no i4 level at all); every (K, codebook) agrees with
           [quant.codebook]; pass 1 first, K2.5-4.5 first inside a pass; names unique and carry the settings stamp
  jobs     every level's jobs (group split; also the tensor split) pass rad-quant-at-home's rxq_quant checks (schema,
           stamp, keys, pins = exactly the implied inputs, requirements) with H_out dense and blocks; ukeys unique;
           H_out inputs at every level (apply_k_min 0); C9 on out_proj / o_proj only (alpha down 0)
  manifest pins carry the manifest's sha256; a missing input is refused
  estimate per-pass rows; blocks cost less than dense; the requirement table puts sub-1 above K2.5
Last line: "LADDER CHECKS n/m PASS"."""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import config as C, ladder as LD, qwen35  # noqa: E402

RQ = sys.argv[1] if len(sys.argv) > 1 else "/workspace/rad-quant-at-home"
jobs = None
if os.path.isdir(os.path.join(RQ, "jobs")):
    sys.path.insert(1, RQ)
    import jobs  # noqa: E402
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


cfg = C.load(); st = C.settings_stamp(cfg)
L = LD.levels(cfg)
cnt = {p: sum(1 for v in L if v["pass"] == p) for p in ("1", "2", "2b", "3")}
p1 = sorted((float(v["K"]), v["codebook"]) for v in L if v["pass"] == "1")
want1 = sorted([(k / 4, "i8") for k in range(5, 21)] + [(5.5, "i8"), (6.0, "i8"), (6.5, "i8"), (7.0, "i8")])
order_ok = [v["pass"] for v in L] == sorted([v["pass"] for v in L], key=["1", "2", "2b", "3"].index) and \
    all(2.5 <= float(L[i]["K"]) <= 4.5 for i in range(9))
names = [v["name"] for v in L]
report("levels", len(L) == 50 and cnt == {"1": 20, "2": 23, "2b": 0, "3": 7} and p1 == want1 and order_ok and
       len(set(names)) == 50 and not any(v["codebook"] in ("i4", "mul1") for v in L) and all(v["codebook"] == "i8" for v in L if float(v["K"]) <= 1) and all(f"-{st[:8]}-" in n for n in names) and all(float(v["K"]) < 1 for v in L if v["pass"] == "3"),
       f"{len(L)} levels {cnt}; pass 1 = the coarse subset {p1 == want1}; order {order_ok}; first {names[:3]}")

for mode in ("dense", "blocks"):
    c = C.with_overrides(cfg, **{"inputs.hout.ldlq": mode})
    n = bad = 0; uk = set(); why = []
    hout_all = c9_ok = True
    for lv in LD.levels(c):
        for split in ("group", "tensor"):
            for j in LD.level_jobs(c, lv, "q27", "r1", None, split=split):
                p = j["params"]; n += 1
                p["pins"] = {k: hashlib.sha256(k.encode()).hexdigest() for k in p["pins"]}
                p["stamp"] = LD.rqah_stamp(p)
                hout_all &= p["hout"] == "bn1920" and sum("/hout/" in k for k in p["pins"]) == len(p["keys"])
                c9_ok &= sorted(p["c9"]) == sorted(k for k in p["keys"] if qwen35.family(k) in ("ssm_out", "attn_output"))
                if (split, j["ukey"]) in uk: bad += 1; why.append(f"duplicate {j['ukey']}")
                uk.add((split, j["ukey"]))
                if jobs is not None:
                    try:
                        m = jobs.check_params(j["type"], p); jobs.check_inputs(j["type"], p, sorted(p["pins"])); jobs.requirements(m, p)
                    except Exception as e:
                        bad += 1; why.append(str(e)[:100])
    report(f"jobs-{mode}", bad == 0 and hout_all and c9_ok,
           f"{n} jobs (group + tensor splits) {'checked by rqah rxq_quant' if jobs else 'NOT checked (no rqah checkout)'}: "
           f"{bad} refused {why[:2]}; H_out inputs everywhere {hout_all}; C9 writers {c9_ok}")

lv = [v for v in L if v["K"] == 2.5][0]
man = {k: {"sha256": hashlib.sha256(k.encode()).hexdigest(), "size": 1}
       for j in LD.level_jobs(cfg, lv, "q27", "r1", None, layers="0-3") for k in j["params"]["pins"]}
js = LD.level_jobs(cfg, lv, "q27", "r1", man, layers="0-3")
pins_ok = all(p == man[k]["sha256"] for j in js for k, p in j["params"]["pins"].items())
k0 = sorted(man)[0]; man2 = {k: v for k, v in man.items() if k != k0}
try: LD.level_jobs(cfg, lv, "q27", "r1", man2, layers="0-3"); refused = False
except SystemExit as e: refused = "not in the manifest" in str(e)
report("manifest", pins_ok and refused and len(js) == 16, f"{len(js)} jobs of layers 0-3, pins = manifest {pins_ok}, a missing input refused {refused}")

ed, eb = LD.estimate(C.with_overrides(cfg, **{"inputs.hout.ldlq": "dense"})), LD.estimate(C.with_overrides(cfg, **{"inputs.hout.ldlq": "blocks"}))
td, tb = sum(r["quant_h"] for r in ed.values()), sum(r["quant_h"] for r in eb.values())
k = "model.language_model.layers.10.mlp.gate_proj"
report("estimate", set(ed) == {"1", "2", "3"} and tb < td and LD.tensor_vram(cfg, k, 0.5) > LD.tensor_vram(cfg, k, 2.5),
       f"quant card-h dense {td:.0f} / blocks {tb:.0f}; gate VRAM K0.5 {LD.tensor_vram(cfg, k, 0.5):.2f} > K2.5 {LD.tensor_vram(cfg, k, 2.5):.2f} GiB")
print(f"LADDER CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
