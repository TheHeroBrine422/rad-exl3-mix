#!/usr/bin/env python3
"""check_repeat.py WORK SPEC [UNIT ...] -- a quant job run again in fresh processes writes the same tensors (GitHub
issue 1). The spec's job runs once as the reference (`python -m rxq job SPEC -o WORK/ref --device D0`), then:
  rep    the same job again, REPEAT_N times (default 2), one line each
  group  one process per Hessian group of the spec's keys, one after another, into one output dir
  dev    the whole job on every other device of DEVICES, one line each (SKIP with one device)
  conc   the groups dealt round-robin over DEVICES, one process a device, all at once (SKIP with one device)
  share  the groups in two halves, both processes at once on the first device (SKIP with one group)
  seed   control: the spec with seed_offset + 1 must differ from the reference (FAIL when equal: the comparison
         cannot fail)
Equal: every t-*.safetensors and c9-*.safetensors of the reference is in the other output with the same tensor names,
dtypes, shapes, values and safetensors metadata, and the other output has no extra such file; bytes are never compared
(the header bytes vary by process). Environment: DEVICES (default 0) a comma list of `rxq job --device` values, torch
device indices in this process's environment or cpu; nothing here masks devices, the caller restricts the visible cards
(ROCm: ROCR_VISIBLE_DEVICES=<cards>, then DEVICES=0,1,... counts within them). DEVICES=cpu with RXQ_CPU_STANDIN=1 (and
RXQ_EXL3_SRC) is a dry run on the stand-in kernels. PY (default this interpreter) runs the jobs, with this checkout's
pipeline dir first on PYTHONPATH. Every run gets a fresh dir under WORK; its output goes to WORK/<run>.log.
Last line: "REPEAT CHECKS n/m PASS"."""
import json, os, shutil, subprocess, sys, time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PIPE)
from rxq import qwen35  # noqa: E402

W, SPEC = sys.argv[1:3]
only = sys.argv[3:]
os.makedirs(W, exist_ok=True)
PY = os.environ.get("PY", sys.executable)
DEVS = os.environ.get("DEVICES", "0").split(",")
NREP = int(os.environ.get("REPEAT_N", "2"))
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (PIPE, os.environ.get("PYTHONPATH", "")) if p))
spec = json.load(open(SPEC))
groups = {}                                         # the spec's keys by Hessian group (layer, group index), first seen first
for k in spec["keys"]: groups.setdefault(qwen35.locate(k)[:2], []).append(k)
groups = list(groups.values())
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


def want(u): return not only or u in only


def start(name, keys, out, dev, **over):
    """one `rxq job` process on a sub-spec (keys replaced, fields overridden) -> (name, Popen, log path)"""
    tag = name.replace(" ", ""); sp = f"{W}/spec-{tag}.json"; json.dump(dict(spec, keys=keys, **over), open(sp, "w"))
    log = f"{W}/{tag}.log"
    p = subprocess.Popen([PY, "-m", "rxq", "job", sp, "-o", out, "--device", dev], env=ENV, stdout=open(log, "w"),
                         stderr=subprocess.STDOUT)
    return name, p, log


def fresh(name):
    out = f"{W}/{name}"; shutil.rmtree(out, ignore_errors=True); return out


def run(name, parts):
    """parts [(keys, dev, out, overrides)] all at once; -> (wall s, None or the first failure with its last lines)"""
    t0 = time.time(); ps = [start(f"{name}-{i}" if len(parts) > 1 else name, ks, out, d, **ov) for i, (ks, d, out, ov) in enumerate(parts)]
    bad = None
    for n, p, log in ps:
        if p.wait() and bad is None: bad = f"{n} rc {p.returncode}: " + " | ".join(x.strip() for x in open(log).readlines()[-6:])
    return time.time() - t0, bad


def files(d): return sorted(f for f in os.listdir(d) if f.startswith(("t-", "c9-")) and f.endswith(".safetensors"))


def diff_file(pa, pb):
    """None if equal, else what differs: metadata keys, tensor names, dtype / shape, or 'name n/numel' per tensor"""
    with safe_open(pa, "pt") as x, safe_open(pb, "pt") as y:
        ma, mb = x.metadata() or {}, y.metadata() or {}
        if ma != mb: return "metadata " + ",".join(sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)))
        if sorted(x.keys()) != sorted(y.keys()): return f"tensor names {sorted(x.keys())} vs {sorted(y.keys())}"
        bad = []
        for n in sorted(x.keys()):
            a, b = x.get_tensor(n), y.get_tensor(n); nm = n.rsplit(".", 1)[-1]
            if a.dtype != b.dtype or a.shape != b.shape: bad.append(f"{nm} {a.dtype}{list(a.shape)} vs {b.dtype}{list(b.shape)}"); continue
            ne = a != b
            if a.is_floating_point(): ne &= ~(a.isnan() & b.isnan())      # NaN in the same place is the same value
            if ne.any(): bad.append(f"{nm} {int(ne.sum())}/{a.numel()}")
    return ", ".join(bad) or None


def compare(a, b):
    """-> (files of a equal in b, files of a, the first difference or None: a file of a missing or different in b, or
    a file b has beyond a)"""
    fa, fb = files(a), files(b); eq, first = 0, None
    for f in fa:
        d = diff_file(f"{a}/{f}", f"{b}/{f}") if f in fb else "missing"
        if d is None: eq += 1
        elif first is None: first = f"{f}: {d}"
    extra = sorted(set(fb) - set(fa))
    if extra and first is None: first = f"{len(extra)} extra files ({extra[0]})"
    return eq, len(fa), first


def unit(name, parts, ref, seq=False):
    if seq:                                          # one process after another (group): wall summed
        s = 0.0
        for i, pt in enumerate(parts):
            dt, bad = run(f"{name}-{i}", [pt]); s += dt
            if bad: break
    else: s, bad = run(name, parts)
    if bad: report(name, False, f"job failed: {bad} ({s:.0f} s)"); return
    eq, n, first = compare(ref, parts[0][2])
    report(name, first is None, f"{eq}/{n} files equal to the reference" + (f"; {first}" if first else "") + f" ({s:.0f} s)")


D0 = DEVS[0]; ref = fresh("ref")
s, bad = run("ref", [(spec["keys"], D0, ref, {})])
if bad or not files(ref):
    report("ref", False, f"the reference job {bad or 'wrote no t-*.safetensors file'} ({s:.0f} s)")
    print(f"REPEAT CHECKS {sum(res)}/{len(res)} PASS"); sys.exit(1)
print(f"ref: {len(files(ref))} files, {len(spec['keys'])} tensors in {len(groups)} Hessian groups on device {D0} ({s:.0f} s)", flush=True)

if want("rep"):
    for i in range(1, NREP + 1): unit(f"rep {i}", [(spec["keys"], D0, fresh(f"rep{i}"), {})], ref)
if want("group"):
    out = fresh("group"); unit("group", [(ks, D0, out, {}) for ks in groups], ref, seq=True)
if want("dev"):
    if len(DEVS) < 2: print("SKIP dev: one device in DEVICES", flush=True)
    for d in DEVS[1:]: unit(f"dev {d}", [(spec["keys"], d, fresh(f"dev{d}"), {})], ref)
if want("conc"):
    if len(DEVS) < 2: print("SKIP conc: one device in DEVICES", flush=True)
    else:
        out = fresh("conc")
        unit("conc", [(sum(groups[i::len(DEVS)], []), d, out, {}) for i, d in enumerate(DEVS) if groups[i::len(DEVS)]], ref)
if want("share"):
    if len(groups) < 2: print("SKIP share: one Hessian group in the spec", flush=True)
    else:
        out = fresh("share"); h = (len(groups) + 1) // 2
        unit("share", [(sum(groups[:h], []), D0, out, {}), (sum(groups[h:], []), D0, out, {})], ref)
if want("seed"):
    out = fresh("seed"); s, bad = run("seed", [(spec["keys"], D0, out, {"seed_offset": spec.get("seed_offset", 0) + 1})])
    if bad: report("seed", False, f"job failed: {bad} ({s:.0f} s)")
    else:
        eq, n, first = compare(ref, out)
        report("seed", first is not None, (f"seed_offset + 1 differs: {eq}/{n} files equal; {first}" if first else
                                           "another seed gave equal files: the comparison cannot fail") + f" ({s:.0f} s)")

print(f"REPEAT CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if res and all(res) else 1)
