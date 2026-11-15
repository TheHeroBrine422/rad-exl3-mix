#!/usr/bin/env python3
"""census_hybrid.py [--census STORE_CENSUS] MANIFEST --mix MIXFILE --base BASE --final-root FR [--codebook CB] [--full] [census.py options]

#185 w173: the census of a hybrid candidate (hostd, eval.final_root): a manifest whose trunk lines are
`<weight> final FR/LEVEL/final codebook=CB` (the levels with a READY marker) and pack / donor / act lines (every other level, and
the rest of the mix). Run by hostd in place of census_manifest.py for any unit with a final root (eval.census_virtual_cmd).

  final lines   LEVEL must have FR/LEVEL/READY (its first line `READY`, its second the rollout's result line: level, VERIFIED,
                status READY, file count, bytes, md5 digest), FR/LEVEL/final.md5 (the md5 list the rollout verified: `hash  ./name`
                per file; its md5 must be the marker's digest), and FR/LEVEL/final/ must hold exactly those files, with the
                listed sizes in total and the listed md5 each. Any difference fails the unit (fail closed).
                Cost: the md5 of a level is computed only when its stat signature (size, mtime, inode of every file) differs from
                the one recorded at its last full pass (FR/LEVEL/census.stat.json). A file rewritten in place with its old size
                and mtime is missed by that fast path; --full computes every md5 again.
  pack lines    census_manifest.py (census.py's checks: donors, act keys, shapes, encodings) on the manifest with the final lines
                taken out; the final weights keep their act lines. Together the two parts cover the trunk: each trunk weight is
                either a pack line (checked here) or a final line (checked above).
  The mixfile must be exactly the manifest's lines after its `base` line (the candidate is what the mix says).

Exit 0: CENSUS HYBRID PASS (one summary line). Exit 1: CENSUS HYBRID FAIL, with the reason.
"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS_MANIFEST = os.path.join(HERE, "census_manifest.py")
LEVEL_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
OWN = ("--mix", "--base", "--final-root", "--codebook", "--census")


def die(msg):
    print(f"CENSUS HYBRID FAIL: {msg}")
    sys.exit(1)


def read_lines(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_final(p):
    return len(p) == 4 and p[1] == "final"


def verify_level(fr, lvl, cb, full):
    """Returns (files, how) where how is 'stat' (the recorded signature matched) or 'md5' (recomputed). Dies on any mismatch."""
    root = os.path.join(fr, lvl)
    ready = os.path.join(root, "READY")
    if not os.path.isfile(ready):
        die(f"{lvl}: no READY marker under {root} (not verified: it may not be served from final/)")
    rl = open(ready).read().splitlines()
    if len(rl) != 2 or rl[0] != "READY":
        die(f"{lvl}: READY marker malformed")
    f = rl[1].split("\t")
    if len(f) < 8 or f[0] != lvl or f[4] != "VERIFIED" or f[7] != "READY":
        die(f"{lvl}: READY marker's result line is not a verified READY result")
    n_expect, bytes_expect, digest = int(f[3]), int(f[2]), f[5]
    if not os.path.isfile(os.path.join(root, "final.md5")):
        die(f"{lvl}: no final.md5 under {root} (the md5 list the marker was written from)")
    raw = open(os.path.join(root, "final.md5"), "rb").read()
    if hashlib.md5(raw).hexdigest()[:32] != digest:
        die(f"{lvl}: final.md5 does not match the READY marker's digest")
    want = {}
    for ln in raw.decode().splitlines():
        h, _, name = ln.partition("  ")
        if not name.startswith("./") or len(h) != 32:
            die(f"{lvl}: final.md5 line malformed: {ln[:80]}")
        want[name[2:]] = h
    if len(want) != n_expect:
        die(f"{lvl}: final.md5 lists {len(want)} files, the marker says {n_expect}")
    d = os.path.join(root, "final")
    if not os.path.isdir(d):
        die(f"{lvl}: no final/ under {root}")
    have = {}
    for dp, _, fn in os.walk(d):
        for x in fn:
            p = os.path.join(dp, x)
            if os.path.islink(p) or not os.path.isfile(p):
                die(f"{lvl}: {os.path.relpath(p, d)} is not a regular file")
            have[os.path.relpath(p, d)] = p
    if set(have) != set(want):
        die(f"{lvl}: final/ does not hold the verified files: missing {sorted(set(want) - set(have))[:5]}, "
            f"extra {sorted(set(have) - set(want))[:5]}")
    sig = {}
    for rel, p in have.items():
        st = os.stat(p)
        sig[rel] = [st.st_size, st.st_mtime_ns, st.st_ino]
    total = sum(v[0] for v in sig.values())
    if total != bytes_expect:
        die(f"{lvl}: final/ holds {total} bytes, the marker says {bytes_expect}")
    cache_p = os.path.join(root, "census.stat.json")
    if not full and os.path.isfile(cache_p):
        try:
            c = json.load(open(cache_p))
        except ValueError:
            c = {}
        if c.get("digest") == digest and c.get("files") == sig:
            return len(want), "stat"
    bad = [rel for rel, h in want.items() if md5_file(have[rel]) != h]
    if bad:
        die(f"{lvl}: md5 differs from the verified list for {len(bad)} file(s), e.g. {bad[:3]}")
    tmp = f"{cache_p}.{os.getpid()}.tmp"
    with open(tmp, "w") as cf:
        json.dump({"digest": digest, "files": sig, "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, cf)
    os.replace(tmp, cache_p)
    return len(want), "md5"


def main(argv):
    if not argv or argv[0].startswith("--"):
        print(__doc__); return 2
    manifest, rest = argv[0], []
    o = {"--mix": None, "--base": None, "--final-root": None, "--codebook": "mad", "--census": None}
    full = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in OWN:
            if i + 1 >= len(argv): die(f"{a} needs a value")
            o[a] = argv[i + 1]; i += 2; continue
        if a == "--full":
            full = True; i += 1; continue
        rest.append(a); i += 1
    if not (o["--mix"] and o["--base"] and o["--final-root"] and o["--census"]):
        die("needs --mix MIXFILE --base BASE --final-root FR --census STORE_CENSUS_PY")
    fr, cb, base = os.path.normpath(o["--final-root"]), o["--codebook"], o["--base"]

    m = read_lines(manifest)
    if not m or len(m[0].split()) != 2 or m[0].split()[0] != "base":
        die("the manifest has no `base PATH` first line")
    if m[0].split()[1] != base:
        die(f"the manifest's base {m[0].split()[1]} is not --base {base}")
    body = m[1:]
    if body != read_lines(o["--mix"]):
        die("the manifest's lines after `base` are not the mixfile's (the candidate is not what the mix says)")

    finals, seen = {}, set()
    for ln in body:
        p = ln.split()
        if p[0] in ("act", "add", "drop"):
            continue                                  # mode / add / drop lines name a weight but are not its entry
        if is_final(p):
            w, d, cbt = p[0], os.path.normpath(p[2]), p[3]
            if cbt != f"codebook={cb}":
                die(f"{w}: {cbt} is not codebook={cb}")
            lvl = os.path.basename(os.path.dirname(d))
            if os.path.basename(d) != "final" or os.path.dirname(os.path.dirname(d)) != fr or not LEVEL_RE.match(lvl):
                die(f"{w}: final dir {d} is not {fr}/LEVEL/final")
            if w in seen: die(f"{w}: named twice")
            seen.add(w); finals[w] = lvl
        elif len(p) == 2:
            if p[0] in seen: die(f"{p[0]}: named twice")
            seen.add(p[0])
        else:
            die(f"unknown manifest line: {ln}")

    levels = sorted(set(finals.values()))
    nfiles, modes = 0, {"stat": 0, "md5": 0}
    for lvl in levels:
        n, how = verify_level(fr, lvl, cb, full)
        nfiles += n; modes[how] += 1

    reduced = [ln for ln in body if not is_final(ln.split())]
    tmpd = tempfile.mkdtemp(prefix="census-hybrid-")
    try:
        man_p, mix_p = os.path.join(tmpd, "pack.mix"), os.path.join(tmpd, "pack.mixfile")
        open(mix_p, "w").write("\n".join(reduced) + "\n")
        open(man_p, "w").write(f"base {base}\n" + "\n".join(reduced) + "\n")
        cmd = [sys.executable, "-I", CENSUS_MANIFEST, "--census", o["--census"], man_p, "--mix", mix_p, "--base", base] + rest
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    sys.stdout.write(p.stdout)
    if p.returncode:
        die(f"pack lines: census_manifest rc {p.returncode}")
    n_pack = sum(1 for ln in reduced if len(ln.split()) == 2 and ln.split()[0] != "drop")
    print(f"CENSUS HYBRID PASS: {len(levels)} final levels ({len(finals)} trunk lines; {nfiles} files: "
          f"{modes['stat']} levels by stat signature, {modes['md5']} by md5), {n_pack} pack/donor lines (census_manifest rc 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
