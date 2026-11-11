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
  post     post_scripts at batch 1 = post_script per level, one --src a swjfit call; post-batch at batch 4 (default,
           asym on, swj k_max 4): one swjfit call per <= 4 consecutive swj levels of one pass, every --src once in
           level order, export before and apply after it, scratches distinct, every other line unchanged
Last line: "LADDER CHECKS n/m PASS"."""
import hashlib, json, math, os, re, sys

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

from rxq import post as P  # noqa: E402
args = ("EXPORT", "MODEL", "SWJMETA", "SCRATCH", "OUT"); SW = "python -m rxq.swjfit "
lva = LD.levels(cfg)
def has_swj(c, v): return any(x[0] == "swj" for x in P.plan_level(c, v["K"]))
def srcs(x): return re.findall(r" --src (\S+?)=", x)
def scr(x): return re.search(r" --scratch (\S+) ", x).group(1)
t1 = LD.post_scripts(cfg, lva, *args); s1 = [x for x in t1.splitlines() if x.startswith(SW)]
n1 = sum(has_swj(cfg, v) for v in lva)
report("post", t1 == "".join(LD.post_script(cfg, v, *args) for v in lva) and len(s1) == n1 and all(x.count(" --src ") == 1 for x in s1),
       f"{len(s1)} swjfit calls for {len(lva)} levels ({n1} with swj), at most {max(x.count(' --src ') for x in s1)} --src")


def batch_ok(c, why):
    """the batch-4 text of every level of c against its batch-1 text: every check of post-batch but the scratch pair"""
    lv = LD.levels(c); ps = {v["name"]: v["pass"] for v in lv}; sw = [v["name"] for v in lv if has_swj(c, v)]
    l4 = LD.post_scripts(c, lv, *args, swj_batch=4).splitlines(); s4 = [x for x in l4 if x.startswith(SW)]
    l1 = LD.post_scripts(c, lv, *args).splitlines()
    runs, prev = [], None                       # maximal runs of consecutive swj levels of one pass
    for v in lv:
        if not has_swj(c, v): prev = None; continue
        if prev is None or prev["pass"] != v["pass"]: runs.append(0)
        runs[-1] += 1; prev = v
    want = sum(math.ceil(n / 4) for n in runs)
    one_pass = all(len(srcs(x)) <= 4 and len({ps[n] for n in srcs(x)}) == 1 for x in s4)
    order = [n for x in s4 for n in srcs(x)] == sw
    at = {n: i for i, x in enumerate(l4) if x.startswith(SW) for n in srcs(x)}
    def first(p): return min(i for i, x in enumerate(l4) if x.startswith(p))
    place = all(n in at and first(f"admin.py export --level {n} ") < at[n] < first(f"python -m rxq.post apply EXPORT/{n}/level ") for n in sw)
    scr_u = len({scr(x) for x in s4}) == len(s4)
    rest = sorted(x for x in l4 if not x.startswith(SW)) == sorted(x for x in l1 if not x.startswith(SW))
    why.append(f"{len(s4)} swjfit calls for {len(sw)} of {len(lv)} levels (want {want}), at most "
               f"{max((len(srcs(x)) for x in s4), default=0)} --src; one pass {one_pass}, order {order}, place {place}, "
               f"scratch unique {scr_u}, other lines {rest}")
    return len(s4) == want and one_pass and order and place and scr_u and rest


wb = []; ok4 = batch_ok(cfg, wb)
n4 = sum(math.ceil(sum(has_swj(cfg, v) for v in lva if v["pass"] == p) / 4) for p in dict.fromkeys(v["pass"] for v in lva))
ok4 &= sum(x.startswith(SW) for x in LD.post_scripts(cfg, lva, *args, swj_batch=4).splitlines()) == n4
ca = C.with_overrides(cfg, **{"post.asym.enable": True}); wa = []; oka = batch_ok(ca, wa)
asym_kept = "# asym is per mix" in LD.post_scripts(ca, lva, *args, swj_batch=4)
ck = C.with_overrides(cfg, **{"post.swj.k_max": 4.0}); wk = []; okk = batch_ok(ck, wk)
A, B, D = lva[0], lva[1], lva[2]
sab, sad = (scr([x for x in LD.post_scripts(cfg, b, *args, swj_batch=2).splitlines() if x.startswith(SW)][0]) for b in ([A, B], [A, D]))
report("post-batch", ok4 and oka and asym_kept and okk and sab != sad,
       f"default: {wb[0]} (per-pass ceil {n4}); asym on: {wa[0]}, asym line kept {asym_kept}; swj k_max 4: {wk[0]}; "
       f"scratch [A,B] != [A,C] {sab != sad}")
print(f"LADDER CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
