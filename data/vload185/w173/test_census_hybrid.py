#!/usr/bin/env python3
"""test_census_hybrid.py -- offline checks of census_hybrid.py's final-line part (#185 w173; no GPU, no packs, no radiance).

  python3 -I test_census_hybrid.py      -> ALL PASS / FAILURES

The final-line checks run before any pack line is read, so every failure case here stops in verify_level / the manifest
parse. The passing path of the whole census (with real packs) is run on ds8 (README, "Census"). This file checks:
  pass      READY + final.md5 + final/ agree; the stat signature cache is written, then used (no md5 the second time);
            a rewrite with new content and a new mtime is caught by md5; a rewrite with the same content is accepted.
  fail      no READY; a malformed result line; a digest that is not final.md5's; a file missing / extra / a non-file; a size
            total that is not the marker's; an md5 that differs (old size and mtime kept: only the md5 pass can see it);
            a codebook other than the one asked for; a final dir outside the final root; a mixfile that is not the manifest's
            body; a weight named twice; a manifest with no `base` line.
"""
import importlib.util, io, os, contextlib, json, hashlib, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("census_hybrid", os.path.join(HERE, "census_hybrid.py"))
CH = importlib.util.module_from_spec(spec); spec.loader.exec_module(CH)

W = tempfile.mkdtemp(prefix="w173-census-hybrid-")
FR = os.path.join(W, "finals")
BASE = os.path.join(W, "base.rad")
open(BASE, "w").write("base\n")
ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


def make_level(lvl, files, status="READY", digest=None, nbytes=None, extra_md5=None):
    root = os.path.join(FR, lvl); d = os.path.join(root, "final"); os.makedirs(d, exist_ok=True)
    lines = []
    for name, data in files.items():
        open(os.path.join(d, name), "wb").write(data)
        lines.append(f"{hashlib.md5(data).hexdigest()}  ./{name}")
    raw = ("\n".join(sorted(lines)) + "\n").encode() if extra_md5 is None else extra_md5
    open(os.path.join(root, "final.md5"), "wb").write(raw)
    dg = digest or hashlib.md5(raw).hexdigest()
    nb = nbytes if nbytes is not None else sum(len(v) for v in files.values())
    row = [lvl, "2.5", str(nb), str(len(files)), "VERIFIED", dg, "IDENTICAL 336/336", status, "t", "1", "x.rad"]
    open(os.path.join(root, "READY"), "w").write("READY\n" + "\t".join(row) + "\n")
    return root


def man_for(lvl_dirs, codebook="mad", extra=None):
    """A manifest with one final line per (weight, level dir) and the matching mixfile body."""
    body = [f"blk.{i}.ssm_qkv.weight final {d} codebook={codebook}" for i, d in enumerate(lvl_dirs)] + (extra or [])
    mf = os.path.join(W, "t.mixfile"); mn = os.path.join(W, "t.mix")
    open(mf, "w").write("\n".join(body) + "\n")
    open(mn, "w").write(f"base {BASE}\n" + "\n".join(body) + "\n")
    return mn, mf


def run_main(mn, mf, cb="mad"):
    """census_hybrid.main on the manifest; returns (rc, stdout)."""
    buf = io.StringIO()
    argv = [mn, "--mix", mf, "--base", BASE, "--final-root", FR, "--codebook", cb, "--census", "/nonexistent/census.py"]
    with contextlib.redirect_stdout(buf):
        try:
            rc = CH.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, buf.getvalue()


def level_ok(lvl, full=False):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return CH.verify_level(FR, lvl, "mad", full)
    except SystemExit:
        return None


def level_fail_msg(lvl, full=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            CH.verify_level(FR, lvl, "mad", full); return ""
        except SystemExit:
            return buf.getvalue().strip()


A = {"q-L0.safetensors": b"alpha" * 100, "q-L1.safetensors": b"bravo" * 100, "model.safetensors.index.json": b"{}\n"}

# ---- pass
root = make_level("L-good", A)
r = level_ok("L-good")
check("pass: READY + final.md5 + final/ agree (md5 path, first pass)", r == (3, "md5"), str(r))
check("pass: stat signature cache written", os.path.isfile(os.path.join(root, "census.stat.json")))
r = level_ok("L-good")
check("pass: second pass uses the stat signature (no md5)", r == (3, "stat"), str(r))
p = os.path.join(root, "final", "q-L0.safetensors")
os.utime(p, ns=(1, 1)); os.utime(p, None)                 # a new mtime, same bytes
r = level_ok("L-good")
check("pass: an mtime change with the same bytes is accepted by md5", r == (3, "md5"), str(r))
r = level_ok("L-good")
check("pass: and the cache is then current again", r == (3, "stat"), str(r))
r = level_ok("L-good", full=True)
check("pass: --full recomputes every md5", r == (3, "md5"), str(r))

# ---- fail
make_level("L-noready", A, status="READY"); os.remove(os.path.join(FR, "L-noready", "READY"))
check("fail: no READY marker", "no READY marker" in level_fail_msg("L-noready"), level_fail_msg("L-noready")[:90])

make_level("L-status", A, status="REJECTED")
check("fail: marker's result is not READY", "not a verified READY result" in level_fail_msg("L-status"))

make_level("L-digest", A, digest="0" * 32)
check("fail: marker digest is not final.md5's", "does not match the READY marker's digest" in level_fail_msg("L-digest"))

root = make_level("L-tamper", A)
p = os.path.join(root, "final", "q-L1.safetensors")
st = os.stat(p); data = bytearray(open(p, "rb").read()); data[7] ^= 0xFF
open(p, "wb").write(bytes(data)); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))     # same size, same mtime: only md5 can see it
check("fail: md5 differs with the old size and mtime", "md5 differs" in level_fail_msg("L-tamper"), level_fail_msg("L-tamper")[:90])

root = make_level("L-extra", A); open(os.path.join(root, "final", "stray.bin"), "wb").write(b"x")
check("fail: extra file in final/", "extra" in level_fail_msg("L-extra") and "stray.bin" in level_fail_msg("L-extra"))

root = make_level("L-missing", A); os.remove(os.path.join(root, "final", "q-L0.safetensors"))
check("fail: missing file in final/", "missing" in level_fail_msg("L-missing"))

root = make_level("L-bytes", A, nbytes=12345)
check("fail: total bytes differ from the marker", "bytes" in level_fail_msg("L-bytes"), level_fail_msg("L-bytes")[:90])

root = make_level("L-nofinal", A); os.remove(os.path.join(root, "final.md5"))
check("fail: no final.md5", "no final.md5" in level_fail_msg("L-nofinal"), level_fail_msg("L-nofinal")[:90])

root = make_level("L-link", A)
p = os.path.join(root, "final", "q-L0.safetensors"); os.remove(p); os.symlink(os.path.join(root, "final", "q-L1.safetensors"), p)
check("fail: a symlink in final/", "not a regular file" in level_fail_msg("L-link"))

# manifest-level failures (the level itself is good)
mn, mf = man_for([os.path.join(FR, "L-good", "final")])
rc, out = run_main(mn, mf)
check("manifest: good level: the final part passes and the pack part runs (its census.py path is a stub here)",
      rc == 1 and "pack lines: census_manifest rc" in out and "final/ does not" not in out and "md5 differs" not in out, out.strip()[-160:])
mn, mf = man_for([os.path.join(FR, "L-good", "final")], codebook="mul1")
rc, out = run_main(mn, mf)
check("manifest: codebook differs from --codebook: FAIL", rc == 1 and "is not codebook=mad" in out, out.strip()[:120])
mn, mf = man_for([os.path.join(W, "elsewhere", "L-good", "final")])
rc, out = run_main(mn, mf)
check("manifest: final dir outside the final root: FAIL", rc == 1 and "is not" in out, out.strip()[:120])
mn, mf = man_for([os.path.join(FR, "L-good", "final")])
open(mf, "a").write("blk.0.extra.weight 0\n")
rc, out = run_main(mn, mf)
check("manifest: mixfile is not the manifest's body: FAIL", rc == 1 and "not the mixfile's" in out, out.strip()[:120])
mn, mf = man_for([os.path.join(FR, "L-good", "final")], extra=["blk.0.ssm_qkv.weight 0"])
rc, out = run_main(mn, mf)
check("manifest: weight named twice: FAIL", rc == 1 and "named twice" in out, out.strip()[:120])
mn, mf = man_for([os.path.join(FR, "L-noready", "final")])
rc, out = run_main(mn, mf)
check("manifest: a level without READY: FAIL before any pack line", rc == 1 and "no READY marker" in out, out.strip()[:120])
open(mn, "w").write("\n".join(open(mf).read().splitlines()) + "\n")
rc, out = run_main(mn, mf)
check("manifest: no base line: FAIL", rc == 1 and "base" in out, out.strip()[:120])

print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
