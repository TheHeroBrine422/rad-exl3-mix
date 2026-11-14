#!/usr/bin/env python3
"""iqh.py NAME K CODE [--rows N] -- #177 (w139, lead note 11-13 1:55 PM CST): an IQR-V4-form lm_head donor through
radiance libquant's own trellis quantiser (gptqsim via w142's abq: the quantiser called as rad-convert calls it, values
decoded to bf16), upstream recipes q38-27b-iq*r-v4's head line (output.k<B>):
  codes=uK tcq=L tcq_rows=256 block=1x* scale=f32 transform=fwht128 tcq_gain=G damp=0.01 calib=<head input H> tcq_order=1
  K 2: L 15, G 1.04 (IQR's two-bit trunk form; IQR has no two-bit head) · K 3 / 4 / 5 / 6: L 16, G 1.08 / 1.12 / 1.16 / 1.24
CODE: mad (tcq_code=mad, IQR's code) or i8 (tcq_lut = mul1i8's 2^L values on its int8 grid: the same libquant path
with our code, so libquant mad vs libquant i8 isolates the code). Env GAIN overrides G.
Writes $W/iq/<NAME>-K<K>/wq.safetensors ("weight" f16, vocab order) and scores it with score119.py on w119's fit and
search dumps (perm id) into $W/scores/{fit,search}/<NAME>-K<K>.npz. --rows N: the first N rows only, timing (no score).
Restartable (each step skipped when its output exists). Run on ds8 under a card lock (CUDA = ROCR_VISIBLE_DEVICES).
(w139, #177)"""
import argparse, json, os, struct, subprocess, sys, time
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

Q = "/data/q27"; W = f"{Q}/w139h"; CK = f"{Q}/models/qwen38-27b-bf16"; HS = f"{Q}/challenge/hessians/calib/lm_head.input.safetensors"
ABQ = f"{W}/abqg"; LIBQ = os.environ.get("LIBQ", f"{Q}/rad10-perf/home/quantizers/libquant.so")
FORM = {2: (15, 1.04), 3: (16, 1.08), 4: (16, 1.12), 5: (16, 1.16), 6: (16, 1.24)}
p = argparse.ArgumentParser(); p.add_argument("name"); p.add_argument("K", type=int); p.add_argument("code", choices=["mad", "i8"])
p.add_argument("--rows", type=int, default=0); a = p.parse_args()
D = f"{W}/iq"; os.makedirs(D, exist_ok=True)
log = lambda m: print(f"{time.strftime('%H:%M:%S', time.gmtime())} {m}", flush=True)

# inputs (shared, cached): the head f32, the head-input Hessian as RADGRAM2 (abgptq.py's writer)
src = f"{D}/head.f32"
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: shape = f.get_slice("lm_head.weight").get_shape()
V, C = shape
if not os.path.exists(src):
    with safe_open(f"{CK}/{idx}", "pt") as f: f.get_tensor("lm_head.weight").float().numpy().tofile(src + ".part")
    os.replace(src + ".part", src)
gd = f"{D}/calib"; gram = f"{gd}/gram.r0.output.bin"
if not os.path.exists(gram):
    os.makedirs(gd, exist_ok=True)
    with open(HS, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k, rows = int(meta["k"]), int(meta["rows"])
    with safe_open(HS, "pt") as f: up = f.get_tensor("hessian_upper").float()
    H = torch.zeros(k, k); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up; H = H + H.triu(1).T
    with open(gram + ".part", "wb") as fh:
        fh.write(b"RADGRAM2" + struct.pack("<IIQII", k, 0, rows, 0, 0)); fh.write(H.numpy().astype(np.float32).tobytes())
    os.replace(gram + ".part", gram)

L, G = FORM[a.K]; G = float(os.environ.get("GAIN", G))
opts = [f"codes=u{a.K}", f"tcq={L}", "tcq_rows=256", "block=1x*", "scale=f32", "transform=fwht128", f"tcq_gain={G}",
        "damp=0.01", f"calib={gd}", "tcq_order=1"]
if a.code == "mad": opts.insert(2, "tcq_code=mad")
else:                                               # mul1i8's L-bit table (trellisL.py's mul1_i8 form), int8 integers
    lut = f"{D}/mul1i8-L{L}.txt"
    if not os.path.exists(lut):
        s = np.arange(1 << L, dtype=np.uint64); s16 = np.arange(1 << 16, dtype=np.uint64)
        bs = lambda t: sum((((t * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)) >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(4)).astype(np.float64)
        b16 = bs(s16); q = np.clip(np.round((bs(s) - b16.mean()) / b16.std() / (3.5 / 127)), -127, 127).astype(int)
        open(lut + ".part", "w").write("\n".join(map(str, q)) + "\n"); os.replace(lut + ".part", lut)
    opts.insert(2, f"tcq_lut={lut}")
od = f"{D}/{a.name}-K{a.K}" + (f"-r{a.rows}" if a.rows else ""); os.makedirs(od, exist_ok=True)
N = a.rows or V
inp = src
if a.rows:
    inp = f"{od}/in.f32"
    if not os.path.exists(inp): np.fromfile(src, np.float32, count=N * C).tofile(inp)
ob = f"{od}/out.bf16"
if not os.path.exists(ob):
    log(f"{a.name} K{a.K}: abq gptqsim {N} x {C} {' '.join(opts)}")
    t = time.time()
    r = subprocess.run([ABQ, LIBQ, "gptqsim", "output.weight", str(N), str(C), inp, ob + ".part", *opts], capture_output=True, text=True)
    if r.returncode: sys.exit(f"abq rc {r.returncode}: {r.stderr[-2000:]}")
    if r.stderr.strip(): print(r.stderr.strip()[-2000:], flush=True)
    os.replace(ob + ".part", ob); json.dump({"sec": time.time() - t, "opts": opts, "rows": N}, open(f"{od}/time.json", "w"))
    log(f"{a.name} K{a.K}: {time.time() - t:.0f} s")
Wq = torch.from_numpy(np.fromfile(ob, dtype=np.int16).copy()).view(torch.bfloat16).view(N, C)
Wr = torch.from_numpy(np.fromfile(inp, np.float32, count=N * C)).view(N, C)
e = ((Wq.float() - Wr) ** 2).sum() / (Wr ** 2).sum()
log(f"{a.name} K{a.K}: plain rel. error {e:.4e}")
if a.rows: sys.exit(0)
wq = f"{od}/wq.safetensors"
if not os.path.exists(wq): save_file({"weight": Wq.to(torch.float16).contiguous()}, wq + ".part"); os.replace(wq + ".part", wq)
del Wq, Wr
for s in ("fit", "search"):
    sd = f"{W}/scores/{s}"; o = f"{sd}/{a.name}-K{a.K}.npz"
    if os.path.exists(o): continue
    r = subprocess.run([sys.executable, "-I", f"{W}/code/v3/yaqa/w119/score119.py", f"{Q}/w119/dump/{s}", CK, sd,
                        f"{a.name}-K{a.K}=exl3:wq={wq},perm=id"], capture_output=True, text=True)
    if r.returncode: sys.exit(f"score {s} rc {r.returncode}: {r.stderr[-2000:]}")
    log(f"{a.name} K{a.K}: scored {s}")
