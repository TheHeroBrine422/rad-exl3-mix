#!/usr/bin/env python3
"""test_sidecheck.py ORIG_OURQ PATCH [--real LAYERS OPTION DIR ...] -- CPU self-test of ourq-sidecheck.patch (w01, ranked.md A8).

Runs the top of ourq.py (argument parsing, layer_plan, the new check, the code stamp's argument dict) for the original
and the patched file, with stub modules for torch / safetensors / exllamav3 (nothing quantizes), on synthetic
side-Hessian dirs:
  - the patched file refuses to start when a quantized trunk linear has no --hout / --hout-diag / --hin file, names it,
    and accepts complete dirs, --hin group files, --only subsets and the head (not checked);
  - --side-partial restores the old behaviour (warning only);
  - the stamp's argument dict is the original's in every case (the new flag is left out; the file hash changes the
    stamp anyway, as with every ourq.py edit).
--real: only report what the check says on existing dirs (layers e.g. 0-23, option --hout / --hin / --hout-diag)."""
import io, json, os, shutil, subprocess, sys, tempfile, types, contextlib
from unittest import mock

ORIG, PATCH = sys.argv[1:3]
T = tempfile.mkdtemp(prefix="w01side-")
os.makedirs(f"{T}/p/quantize"); shutil.copy(ORIG, f"{T}/p/quantize/ourq.py")
subprocess.run(["git", "apply", os.path.abspath(PATCH)], cwd=f"{T}/p", check=True)
PAT = f"{T}/p/quantize/ourq.py"
CK = f"{T}/ckpt"; os.makedirs(CK); json.dump({"weight_map": {}}, open(f"{CK}/model.safetensors.index.json", "w"))
open(f"{T}/q.py", "w").write("# stub exllamav3 quantize.py\n")

def stubs():
    m = {}
    for n in ("torch", "safetensors", "safetensors.torch"): m[n] = mock.MagicMock()
    names = ["exllamav3", "exllamav3.modules", "exllamav3.modules.quant", "exllamav3.modules.quant.exl3_lib",
             "exllamav3.modules.quant.exl3_lib.quantize"]
    for n in names: m[n] = types.ModuleType(n)
    for a, b in zip(names, names[1:]): setattr(m[a], b.rsplit(".", 1)[1], m[b])
    q = m[names[-1]]; q.__file__ = f"{T}/q.py"; q.LDLQ_DRIFT = {1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004}
    q.quantize_exl3 = q.refit_scales = q.g_scale_gss = lambda *x, **k: None; q.ldlq_drift = lambda K: 1.0
    return m

END = "STAMP, STAMP_SRC, STAMP_ARGS = _stamp()\n"
def top(path, argv):
    """exec ourq.py up to the stamp; -> (exit message or None, printed text, the stamp's argument dict)"""
    src = open(path).read(); src = src[:src.index(END) + len(END)]
    src = src.replace("_warm_refit(dev)                                   # load", "pass  # load")   # test-only: no device
    g = {"__name__": "ourq_top", "__file__": path}; out = io.StringIO()
    with mock.patch.dict(sys.modules, stubs()), mock.patch.object(sys, "argv", ["ourq.py"] + argv), \
         mock.patch.dict(os.environ, {"RXM_CKPT": CK}), contextlib.redirect_stdout(out):
        try: exec(compile(src, path, "exec"), g)
        except SystemExit as e: return str(e), out.getvalue(), None
    return None, out.getvalue(), g["STAMP_ARGS"]

L = "model.language_model.layers"
def plan_keys(lo, hi):            # (group, key) of the 0.8B / 27B layer plan, as ourq.py's layer_plan
    r = []
    for l in range(lo, hi + 1):
        p = f"{L}.{l}"; full = l % 4 == 3
        r += [(f"{p}.input", k) for k in ([f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"] if full
                                          else [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"])]
        r += [(f"{p}.self_attn.o_proj.input" if full else f"{p}.linear_attn.out_proj.input",
               f"{p}.self_attn.o_proj" if full else f"{p}.linear_attn.out_proj")]
        r += [(f"{p}.post_attention.input", f"{p}.mlp.gate_proj"), (f"{p}.post_attention.input", f"{p}.mlp.up_proj"),
              (f"{p}.mlp.down_proj.input", f"{p}.mlp.down_proj")]
    return r

if "--real" in sys.argv:
    a = sys.argv[sys.argv.index("--real") + 1:]; lay = a[0]
    for opt, d in zip(a[1::2], a[2::2]):
        err, txt, _ = top(PAT, ["-K", "3", "-o", f"{T}/o", "--layers", lay, opt, d])
        n = sum(1 for f in os.listdir(d) if f.endswith(".safetensors"))
        print(f"{'REFUSED' if err else 'ok     '}  {opt} {d} ({n} files): {err or 'every quantized linear has a file'}")
    shutil.rmtree(T); sys.exit(0)

def mk(name, keys):
    d = f"{T}/{name}"; os.makedirs(d)
    for k in keys: open(f"{d}/{k}.safetensors", "w").close()
    return d
K4 = plan_keys(0, 3)
full = mk("full", [k for _, k in K4]); miss = mk("miss", [k for _, k in K4 if not k.endswith("2.mlp.down_proj")])
hgrp = mk("hgrp", sorted({g for g, _ in K4}))                                         # collect.py YAQA_HIN layout
hkey = mk("hkey", [k for _, k in K4])                                                  # yaqaB.py layout
hpart = mk("hpart", [g for g in sorted({g for g, _ in K4}) if not g.endswith("1.post_attention.input")])
empty = mk("empty", [])
only = f"{T}/only.txt"; open(only, "w").write("\n".join(k for _, k in K4 if ".2." not in k) + "\n")
B = ["-K", "3", "-o", f"{T}/o", "--layers", "0-3"]
cases = [  # name, args, patched must refuse?, text the refusal must contain
    ("no side options", [], False, None),
    ("--hout complete", ["--hout", full], False, None),
    ("--hout one missing", ["--hout", miss], True, "layers.2.mlp.down_proj"),
    ("--hout one missing + --side-partial", ["--hout", miss, "--side-partial"], False, None),
    ("--hout-diag one missing", ["--hout-diag", miss], True, "--hout-diag: 1 of 25"),
    ("--hout-diag complete", ["--hout-diag", full], False, None),
    ("--hin group files (collect.py)", ["--hin", hgrp], False, None),
    ("--hin per-key files (yaqaB.py)", ["--hin", hkey], False, None),
    ("--hin one group missing", ["--hin", hpart], True, "--hin: 2 of 25"),
    ("--hout one missing, --only without it", ["--hout", miss, "--only", only], False, None),
    ("--hout empty dir", ["--hout", empty], True, "--hout: 25 of 25"),
    ("--only-head, --hout empty (head not checked)", ["--hout", empty, "--only-head", "--head"], False, None),
    ("--hout one missing, --layers 0-1 (outside)", ["--hout", miss, "--layers", "0-1"], False, None),
]
ok = True
for name, args, refuse, txt in cases:
    eo, _, ko = top(ORIG, B + [x for x in args if x != "--side-partial"]); ep, out, kp = top(PAT, B + args)   # (the original lacks the flag)
    if eo is not None: print(f"FAIL  {name}: the ORIGINAL exits ({eo})"); ok = False; continue
    good = (ep is not None) == refuse and (txt is None or txt in (ep or "")) and (refuse or ko == kp)
    if "--side-partial" in args: good &= "warning" in out and "--side-partial" in out
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: {'refuses: ' + ep if ep else 'runs; stamp arguments ' + ('equal' if ko == kp else 'DIFFER')}"
          + (f"; prints: {out.strip().splitlines()[-1]}" if "--side-partial" in args else ""))
print("ALL PASS" if ok else "SOME FAILED"); shutil.rmtree(T)
