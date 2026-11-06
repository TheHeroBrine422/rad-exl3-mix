#!/usr/bin/env python3
"""test_cpu.py SO FRAC27 FRAC08 INT08 -- CPU tests of asym27's two changes to v3/sib27 (run on the 4070 box, CPU only:
CUDA_VISIBLE_DEVICES= nice -n 19, 4 threads). SO = the plugin's exl3i oracle (gcc -O2 -ffp-contract=off -shared -fPIC
exl3i_decode.c exl3_decode.c exl3_layout.c -lm); FRAC27 = an ourq export of real 27B tensors at a fractional rate with
--save-wq (hconv/q-bn128: layers 10 / 39 / 50, K2.5, i8); FRAC08 = a fractional 0.8B level (K1.5, i8); INT08 = an
integer 0.8B level (q-k3base, K3, i8).
  A  deq.py oracle (W_hat bit for bit vs exl3i_decode_what_rows in mul1 and i8, W vs exl3i_decode_rows) on FRAC27's
     siblings and writers, FRAC08 and INT08
  B  synthetic trellises at 16 rates (sub-1 K0.125 .. K8, odd eighths): decode_hat vs the oracle, both codebooks
  C  integer K: the new decode_hat = v3/sib27/deq.py's, bit for bit (random m = 16 .. 128 and INT08's tensors)
  D  deq.py check FRAC27 vs its --save-wq files (the quantizer's own weights): worst rel < 2e-3, wrong codebook ~1e-2,
     except the tensors whose scale refit took ourq's CPU fallback (OURQ_REFIT_CPU_FALLBACK, logged "refit on the
     CPU" in FRAC27's quant log FRAC27.log): there --save-wq's output scales differ from the shard's (found here,
     2026-11-06), so those must instead fall to wq's fp16 storage floor (< 3e-4) once one factor per row and per
     column is fitted out (scalecheck.py): the shard, which radiance serves, is what deq.py decodes
  E  sibmom27's save_lean = safetensors' save_file (tensors, dtypes, shapes, metadata)
Prints one line per test and ALL PASS / FAILED."""
import ctypes, importlib.util, os, re, subprocess, sys, tempfile
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f"{HERE}/../../..")
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
SO, FRAC27, FRAC08, INT08 = sys.argv[1:5]
sys.path.insert(0, HERE)
import deq                                                    # asym27's
spec = importlib.util.spec_from_file_location("deq_old", f"{REPO}/v3/sib27/deq.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
res = {}


def run(*args):
    p = subprocess.run([sys.executable, f"{HERE}/deq.py", *args], capture_output=True, text=True, env=dict(os.environ, DEV="cpu"))
    out = p.stdout + p.stderr
    print(out.rstrip(), flush=True)
    return out


# A
k27 = ",".join(f"model.language_model.layers.{x}" for x in (
    "10.linear_attn.in_proj_z", "10.linear_attn.out_proj", "10.mlp.down_proj", "39.self_attn.k_proj",
    "39.self_attn.o_proj", "50.mlp.up_proj"))
ok = all("DEQ ORACLE: bit-exact" in run("oracle", lv, SO, sel) for lv, sel in ((FRAC27, k27), (FRAC08, "6"), (INT08, "6")))
res["A oracle (27B K2.5, 0.8B K1.5, 0.8B K3)"] = ok


# B
class View(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_int64), ("cols", ctypes.c_int64), ("m", ctypes.c_int), ("cb", ctypes.c_int),
                ("record", ctypes.c_void_p), ("suh", ctypes.c_void_p), ("suh1", ctypes.c_void_p), ("svh", ctypes.c_void_p)]
lib = ctypes.CDLL(SO)
lib.exl3i_decode_what_rows.argtypes = [ctypes.POINTER(View), ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
g = torch.Generator().manual_seed(0); bad = []
for m in (2, 4, 6, 8, 10, 14, 16, 18, 24, 34, 40, 48, 64, 98, 126, 128):
    kt, nt = 8, 6
    tr = torch.randint(-32768, 32768, (kt, nt, m), generator=g, dtype=torch.int32).to(torch.int16)
    rec = np.ascontiguousarray(tr.numpy().view(np.uint32).reshape(kt, nt, m // 2).transpose(1, 0, 2))
    for cbn, cb in (("mul1", 0), ("i8", 1)):
        v = View(nt * 16, kt * 16, m, cb, rec.ctypes.data, None, None, None)
        ref = np.empty((nt * 16, kt * 16), np.float32)
        rc = lib.exl3i_decode_what_rows(ctypes.byref(v), 0, nt * 16, ref.ctypes.data)
        mine = deq.decode_hat(tr, deq.lut(cbn)).numpy().T
        n = int((mine.view(np.uint32) != ref.view(np.uint32)).sum()) if rc == 0 else -1
        if n: bad.append((m, cbn, rc, n))
print(f"B synthetic rates: {16 * 2} cases, differing {bad}", flush=True)
res["B synthetic rates incl. sub-1"] = not bad

# C
bad = []
for m in range(16, 129, 16):
    tr = torch.randint(-32768, 32768, (4, 5, m), generator=g, dtype=torch.int32).to(torch.int16)
    for cbn in ("mul1", "i8"):
        if not torch.equal(deq.decode_hat(tr, deq.lut(cbn)), old.decode_hat(tr, old.lut(cbn))): bad.append((m, cbn))
lv = deq.Level(INT08, dev="cpu"); nk = 0
for k in lv.keys()[:12]:
    tr = lv._t(f"{k}.trellis"); nk += 1
    if not torch.equal(deq.decode_hat(tr, lv.table), old.decode_hat(tr, old.lut(lv.codebook))): bad.append(k)
    if not torch.equal(lv.get(k), old.Level(INT08, dev="cpu").get(k)): bad.append(k + " (W)")
print(f"C integer K: 8 random rates x 2 codebooks + {nk} real tensors (W_hat and W), differing {bad}", flush=True)
res["C integer path = sib27 deq"] = not bad

# D
log = open(f"{FRAC27}.log").read().splitlines() if os.path.exists(f"{FRAC27}.log") else []
fb = {re.split(r"\s+", log[i + 1].strip())[0] for i, l in enumerate(log[:-1]) if "refit on the CPU" in l}
out = run("check", FRAC27, f"{FRAC27}/wq")
rows = {k: (float(r), float(o)) for k, r, o in re.findall(r"^(\S+)\s+rel ([0-9.e+-]+) \(max elem [0-9.e+-]+\); with the other codebook ([0-9.e+-]+)", out, re.M)}
rest = {k: v for k, v in rows.items() if k not in fb}
worst = max(r for r, _ in rest.values()); oth = min(o for _, o in rows.values())
sc = subprocess.run([sys.executable, f"{HERE}/scalecheck.py", FRAC27, *sorted(fb)], capture_output=True, text=True).stdout if fb else ""
print(sc.rstrip(), flush=True)
floor = [float(x) for x in re.findall(r"-> ([0-9.e+-]+) after", sc)]
ok = len(rows) > 0 and worst < 2e-3 and oth > 10 * worst and len(floor) == len(fb) and all(x < 3e-4 for x in floor)
res["D frac level vs --save-wq"] = ok
print(f"D: {len(rest)} tensors worst rel {worst:.2e} (other codebook >= {oth:.1e}); {len(fb)} CPU-refit-fallback tensors "
      f"after row / column factors {max(floor, default=0):.2e}", flush=True)

# E
src = open(f"{HERE}/sibmom27.py").read()
fn = src[src.index("def save_lean("):src.index("\n\n\ndef save(")]
ns = {"json": __import__("json"), "os": os, "torch": torch}; exec(fn, ns)
with tempfile.TemporaryDirectory() as d:
    tot = {s: torch.randn(300 + i, 300 + i, dtype=torch.float64, generator=g) for i, s in enumerate("HCN")}
    tot = {s: v[:300, :300].contiguous() for s, v in tot.items()}
    md = {"n": "1234", "mode": "sib", "wq": "x", "kernels": "{\"a\": 1}", "level": "None"}
    ns["save_lean"](f"{d}/a.safetensors", tot, md)
    save_file({s: v.float().contiguous() for s, v in tot.items()}, f"{d}/b.safetensors", metadata=md)
    with safe_open(f"{d}/a.safetensors", "pt") as A, safe_open(f"{d}/b.safetensors", "pt") as B:
        same = A.metadata() == B.metadata() and sorted(A.keys()) == sorted(B.keys()) and \
            all(torch.equal(A.get_tensor(s), B.get_tensor(s)) and A.get_tensor(s).dtype == torch.float32 for s in "HCN")
print(f"E save_lean vs save_file: {'identical tensors and metadata' if same else 'DIFFER'}", flush=True)
res["E save_lean = save_file"] = same

for k, v in res.items(): print(f"{'PASS' if v else 'FAIL'}  {k}")
print("ALL PASS" if all(res.values()) else "FAILED")
sys.exit(0 if all(res.values()) else 1)
