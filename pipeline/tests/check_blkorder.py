#!/usr/bin/env python3
"""check_blkorder.py FORK_DENSE FORK_BLOCKS [WORK] -- #117 route B in rxq (rxq/blkorder.py, quant.block_order) on the CPU
stand-in (rxq/cpukern.py; the fork's real quantize.py), tiny attention layer 3 with the real residual width (readers
q / k / v / gate / up read 5120 columns; writers o / down write 5120 rows), V3 settings at K2.5 i8 (H_out, tile CD on
q_proj, C9 on o_proj), H_out dense (FORK_DENSE) and blocks (FORK_BLOCKS, inputs.hout.ldlq = blocks):
  noop     block_order on with the identity order == block_order off, bit for bit (trellis, suh, svh, C9 scales), with
           the two-sided LDLQ and tile CD running inside the wrappers
  order    w82's consensus order: every reader's codes change, every writer is bit-identical to the plain run (route B
           touches readers only), the log marks the readers (block_order) and nothing else
  decode   every stored tensor decodes (rxq.deq) to the weight the quantizer reports (wq) to f16 rounding: the
           permuted LDLQ's rows and tile indices went back to natural order
  cd       tile CD on q_proj ran inside the permutation (its record exists, ratio <= 1: CD only accepts changes that
           lower its objective, so a metric / weight mismatch would show as a decode or objective failure)
  stamp    the settings stamp changes with enable / the order; enable = false with any order is refused nowhere, and a
           non-permutation is refused by config.validate
Last line: "BLKORDER CHECKS n/m PASS"."""
import json, os, re, subprocess, sys, tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PIPE)
from rxq import config as C, deq, job, qwen35  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "--run":
    job.run_quant(json.load(open(sys.argv[2])), sys.argv[3], device="cpu", save_wq=True)
    sys.exit(0)

FD, FB = sys.argv[1], sys.argv[2]
W = sys.argv[3] if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="w86blk-")
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


SH = {"attn_qg": (256, 5120), "attn_k": (128, 5120), "attn_v": (128, 5120), "attn_output": (5120, 256),
      "ffn_gate": (256, 5120), "ffn_up": (256, 5120), "ffn_down": (5120, 384)}
l = 3
g = torch.Generator().manual_seed(117)
ck, hs, ho = f"{W}/ckpt", f"{W}/hess", f"{W}/hout/bn1920"
for d in (ck, hs, ho): os.makedirs(d, exist_ok=True)
tens, keys = {}, []
for hname, ks in qwen35.layer_plan(l):
    k_in = SH[qwen35.family(ks[0])][1]
    if not os.path.exists(f"{hs}/{hname}.safetensors"):
        A = torch.randn(1024, k_in, generator=g) * torch.linspace(0.2, 3.0, k_in)[torch.randperm(k_in, generator=g)]
        H = (A.T @ A).float() + torch.diag(torch.rand(k_in, generator=g) * 50); iu = torch.triu_indices(k_in, k_in)
        save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hs}/{hname}.safetensors", metadata={"k": str(k_in), "rows": "1024"})
        del A, H, iu
    for key in ks:
        n, k = SH[qwen35.family(key)]
        tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
        save_file({"hout_diag": (torch.rand(n, generator=g) + 0.1).float()}, f"{ho}/{key}.safetensors")
        keys.append((hname, key))
save_file(tens, f"{ck}/model.safetensors")

base = open(os.path.join(PIPE, "config", "v3.toml")).read()
sec = re.search(r"\[quant\.block_order\]\n(?:.*\n)*?enable = (?:false|true)\nstored_order = \[[^\]]*\][^\n]*\n", base)
assert sec, "v3.toml: [quant.block_order] not found"
base = base.replace(sec.group(0), "[quant.block_order]\nenable = false\nstored_order = " + json.dumps(list(range(40))) + "\n")
base = re.sub(r'^ldlq = "(dense|blocks)"', 'ldlq = "dense"', base, flags=re.M)        # variant() sets both switches
P = json.load(open(os.path.join(PIPE, "..", "v3", "e3", "settle", "w84", "order82.json")))["stored_order"]


BSEC = "[quant.block_order]\nenable = false\nstored_order = " + json.dumps(list(range(40))) + "\n"


def variant(order, blocks):
    t = base
    if order is not None:
        t = t.replace(BSEC, "[quant.block_order]\nenable = true\nstored_order = " + json.dumps(order) + "\n")
    if blocks: t = t.replace('ldlq = "dense"', 'ldlq = "blocks"')
    return t


runs = {}
for mode, fork in (("dense", FD), ("blocks", FB)):
    for name, order in (("plain", None), ("ident", list(range(40))), ("order", P)):
        tp = f"{W}/{mode}-{name}.toml"; open(tp, "w").write(variant(order, mode == "blocks"))
        cfg = C.load(tp)
        out = f"{W}/out-{mode}-{name}"; ok = True
        for gi, (hname, ks) in enumerate(qwen35.layer_plan(l)):
            spec = {"type": "rxq.quant", "version": 1, "level": "t-k2.5-i8", "K": 2.5, "codebook": "i8", "seed_offset": 0,
                    "keys": ks, "settings_stamp": C.settings_stamp(cfg), "config": tp,
                    "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": ho}}
            sp = f"{W}/spec-{mode}-{name}-{gi}.json"; json.dump(spec, open(sp, "w"))
            env = dict(os.environ, RXQ_CPU_STANDIN="1", RXQ_EXL3_SRC=f"{fork}/exllamav3/exllamav3_ext", PYTHONPATH=PIPE, OMP_NUM_THREADS="2")
            r = subprocess.run([sys.executable, __file__, "--run", sp, out], env=env, capture_output=True, text=True)
            if r.returncode: ok = False; print(r.stdout[-1500:], r.stderr[-3000:]); break
        runs[(mode, name)] = (out, ok, C.settings_stamp(cfg))
        report(f"run-{mode}-{name}", ok, f"layer 3, {len(keys)} tensors, K2.5 i8, H_out {mode}, block order {name}")


def load(out, key):
    with safe_open(job.tensor_file(out, key), "pt") as f: t = {n.rsplit(".", 1)[1]: f.get_tensor(n) for n in f.keys()}
    c9 = None
    if os.path.exists(job.c9_path(out, key)):
        with safe_open(job.c9_path(out, key), "pt") as f: c9 = f.get_tensor("c")
    return t, c9


def same(a, b):
    (ta, ca), (tb, cb) = a, b
    return all(torch.equal(ta[x], tb[x]) for x in ta) and ((ca is None and cb is None) or (ca is not None and cb is not None and torch.equal(ca, cb)))


table = torch.from_numpy(deq.codebook_lut("i8").astype(np.float32))
for mode in ("dense", "blocks"):
    (op, okp, sp_), (oi, oki, si), (oo, oko, so) = runs[(mode, "plain")], runs[(mode, "ident")], runs[(mode, "order")]
    if not (okp and oki and oko): continue
    noop = [same(load(op, k), load(oi, k)) for _, k in keys]
    report(f"noop-{mode}", all(noop), f"identity order == plain on {sum(noop)}/{len(keys)} tensors (trellis, suh, svh, C9)")
    rd = [k for _, k in keys if qwen35.family(k) in qwen35.READERS]
    chg = {k: not same(load(op, k), load(oo, k)) for _, k in keys}
    recs = {json.loads(x)["key"]: json.loads(x) for x in open(f"{oo}/log.jsonl")}
    marks = {k: bool(recs[k].get("block_order")) for _, k in keys}
    report(f"order-{mode}", all(chg[k] for k in rd) and not any(chg[k] for k in chg if k not in rd) and all(marks[k] == (k in rd) for k in marks),
           f"readers changed {[k.rsplit('.', 1)[1] for k in rd if chg[k]]}, writers identical "
           f"{[k.rsplit('.', 1)[1] for k in chg if k not in rd and not chg[k]]}, log marks {sorted(k.rsplit('.', 1)[1] for k in marks if marks[k])}")
    worst = 0.0
    for _, k in keys:
        t, _ = load(oo, k)
        Wd = deq.weight(t["trellis"], t["suh"], t["svh"], 2.5, table)
        with safe_open(f"{oo}/wq/{k}.safetensors", "pt") as f: wq = f.get_tensor("weight").float().T
        worst = max(worst, float((Wd - wq).norm() / wq.norm()))
    report(f"decode-{mode}", worst < 2e-3, f"block-ordered level: max |W_decoded - wq| / |wq| = {worst:.2e}")
    cd = [json.loads(x) for x in open(f"{oo}/tilecd.jsonl")] if os.path.exists(f"{oo}/tilecd.jsonl") else []
    report(f"cd-{mode}", len(cd) == 1 and cd[0]["key"].endswith("q_proj") and cd[0]["ratio"] <= 1.0 and cd[0].get("hout"),
           f"tile CD inside the permutation: {[(c['key'].rsplit('.', 1)[1], round(c['ratio'], 4)) for c in cd]}; proxy plain / ordered "
           f"q_proj {json.loads(open(f'{op}/log.jsonl').readline())['proxy_err']:.4f} / {recs[rd[0]]['proxy_err']:.4f}")
    report(f"stamp-{mode}", len({sp_, si, so}) == 3, f"settings stamps plain {sp_} / ident {si} / order {so}")
try:
    C.with_overrides(C.load(), **{"quant.block_order.stored_order": list(range(39)) + [0]}); bad = False
except ValueError as e:
    bad = "permutation" in str(e)
report("validate", bad, "a non-permutation order is refused")
print(f"BLKORDER CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
