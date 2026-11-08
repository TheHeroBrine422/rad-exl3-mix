#!/usr/bin/env python3
"""check_blkorder84.py FORK [WORK] -- rxq's #117 block order (rxq/blkorder.py, quant.block_order on) == w84's route B
(v3/e3/settle/w84/blkorder84.py installed on rxq's engine exactly as w84q.py installs it, quant.block_order off), bit
for bit, on the CPU stand-in (rxq/cpukern.py; the fork's real quantize.py at FORK), a tiny attention layer 3 with the real
residual width (readers q / k / v / gate / up read 5120 columns), K2.5 i8, under w84's p0 settings: H_out off (the plain
controls, `rxq levelargs`), tile CD off (w84.sh adds quant.tile_cd.enable=false); also with tile CD on (CD inside both
wrappers). The codes are the stand-in's, not the card's: what this proves is that rxq's port reorders exactly as the
module that made w84's p0perm / p1perm tensors (`tools/blkorder27.sh` re-makes p0perm's 9 noop-gate readers on card 0
with rxq and compares them bit for bit). Compared: trellis / suh / svh / mul1 of every tensor (the metadata differs by
design: the settings stamp holds quant.block_order). Last line: "BLKORDER84 CHECKS n/m PASS"."""
import json, os, re, subprocess, sys, tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
W84 = os.path.normpath(os.path.join(PIPE, "..", "v3", "e3", "settle", "w84"))
sys.path.insert(0, PIPE)
from rxq import config as C, job, qwen35  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "--run":          # child: one job; argv[4] = "w84" installs blkorder84 as w84q.py
    spec = json.load(open(sys.argv[2]))
    if sys.argv[4] == "w84":
        sys.path.insert(0, W84)
        import blkorder84
        from rxq import engine
        _setup = engine.setup

        def setup(*a, **kw):
            Q = _setup(*a, **kw); blkorder84.install(Q, os.path.join(W84, "order82.json")); return Q
        engine.setup = setup
    job.run_quant(spec, sys.argv[3], device="cpu")
    sys.exit(0)

FORK = sys.argv[1]
W = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="w86b84-")
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


SH = {"attn_qg": (256, 5120), "attn_k": (128, 5120), "attn_v": (128, 5120), "attn_output": (5120, 256),
      "ffn_gate": (256, 5120), "ffn_up": (256, 5120), "ffn_down": (5120, 384)}
l = 3; g = torch.Generator().manual_seed(84)
ck, hs = f"{W}/ckpt", f"{W}/hess"
for d in (ck, hs): os.makedirs(d, exist_ok=True)
tens, keys = {}, []
for hname, ks in qwen35.layer_plan(l):
    k_in = SH[qwen35.family(ks[0])][1]
    A = torch.randn(1024, k_in, generator=g) * torch.linspace(0.2, 3.0, k_in)[torch.randperm(k_in, generator=g)]
    H = (A.T @ A).float() + torch.diag(torch.rand(k_in, generator=g) * 50); iu = torch.triu_indices(k_in, k_in)
    save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hs}/{hname}.safetensors", metadata={"k": str(k_in), "rows": "1024"})
    del A, H, iu
    for key in ks:
        n, k = SH[qwen35.family(key)]
        tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16); keys.append(key)
save_file(tens, f"{ck}/model.safetensors")

base = open(os.path.join(PIPE, "config", "v3.toml")).read()
order = json.load(open(os.path.join(W84, "order82.json")))["stored_order"]
p0 = json.load(open(os.path.join(PIPE, "..", "v3", "e3", "settle", "data", "w84", "p0-blkorder84.json")))[0]["stored_order"]
cfg0 = C.load(os.path.join(PIPE, "config", "v3.toml"))
report("order", C.block_order(cfg0) == order == p0, f"v3.toml's stored order == w84's order82.json == the p0 run's record: "
       f"{C.block_order(cfg0) == order} / {order == p0}; enabled by default {C.block_order(cfg0) is not None}")


def variant(bo, cd):
    t = re.sub(r"(\[quant\.block_order\]\n(?:#.*\n)*)enable = (true|false)", lambda m: m.group(1) + f"enable = {'true' if bo else 'false'}", base)
    t = re.sub(r"(\[inputs\.hout\]\n(?:.*\n)*?)enable = true", r"\1enable = false", t, count=1)      # the plain controls
    if not cd: t = re.sub(r"(\[quant\.tile_cd\]\n(?:#.*\n)*)enable = true", r"\1enable = false", t)
    return t


for cd in (False, True):
    outs = {}
    for who, bo in (("rxq", True), ("w84", False)):
        tp = f"{W}/{who}-cd{int(cd)}.toml"; open(tp, "w").write(variant(bo, cd)); cfg = C.load(tp)
        assert (C.block_order(cfg) is not None) == bo and not C.hout_applies(cfg, 2.5) and bool(cfg["quant"]["tile_cd"]["enable"]) == cd
        out = f"{W}/out-{who}-cd{int(cd)}"; outs[who] = out
        for gi, (hname, ks) in enumerate(qwen35.layer_plan(l)):
            spec = {"type": "rxq.quant", "version": 1, "level": "t", "K": 2.5, "codebook": "i8", "seed_offset": 0, "keys": ks,
                    "settings_stamp": C.settings_stamp(cfg), "config": tp,
                    "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": None}}
            sp = f"{W}/spec-{who}-{cd}-{gi}.json"; json.dump(spec, open(sp, "w"))
            env = dict(os.environ, RXQ_CPU_STANDIN="1", RXQ_EXL3_SRC=f"{FORK}/exllamav3/exllamav3_ext", PYTHONPATH=PIPE, OMP_NUM_THREADS="2")
            r = subprocess.run([sys.executable, __file__, "--run", sp, out, who], env=env, capture_output=True, text=True)
            if r.returncode: print(r.stdout[-1000:], r.stderr[-2500:]); break
    eq, marks = [], []
    for key in keys:
        fa, fb = job.tensor_file(outs["rxq"], key), job.tensor_file(outs["w84"], key)
        if not (os.path.exists(fa) and os.path.exists(fb)): eq.append(False); continue
        with safe_open(fa, "pt") as a, safe_open(fb, "pt") as b:
            eq.append(sorted(a.keys()) == sorted(b.keys()) and all(torch.equal(a.get_tensor(n), b.get_tensor(n)) for n in a.keys()))
    recs = {json.loads(x)["key"]: json.loads(x) for x in open(f"{outs['rxq']}/log.jsonl")} if os.path.exists(f"{outs['rxq']}/log.jsonl") else {}
    rd = [k for k in keys if qwen35.family(k) in qwen35.READERS]
    report(f"w84-cd{int(cd)}", all(eq) and len(eq) == len(keys) and all(recs.get(k, {}).get("block_order") for k in rd),
           f"tile CD {'on' if cd else 'off'}: rxq block order == blkorder84 on {sum(eq)}/{len(keys)} tensors (readers {len(rd)}, "
           f"writers {len(keys) - len(rd)}); rxq's log marks the readers {all(recs.get(k, {}).get('block_order') for k in rd)}")
print(f"BLKORDER84 CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
