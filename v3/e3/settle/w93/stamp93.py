#!/usr/bin/env python3
"""stamp93.py RXM_REV FORK_DIR FORK_REV [--standin] -- w93 (review item 7): rxq's settings stamp and code stamp of a
rad-exl3-mix commit and an exllamav3 fork commit, computed from git objects (no checkout, no kernel build).

The code stamp replicates pipeline/rxq/engine.py code_stamp(Q) for a GPU run (or the CPU stand-in with --standin):
sha256 over (basename, bytes) of, in order: pipeline/rxq/*.py (sorted paths); the fork's
exllamav3/modules/quant/exl3_lib/quantize.py (Q.__file__); RXQ_EXL3_SRC = the fork's exllamav3/exllamav3_ext
(*.cu, *.cuh, *.cpp, *.h, *.py, recursive, sorted paths); the run-time kernel dirs v3/e3/{exl3frac, exl3cb, exl3frac2}
(every *.* file, sorted), including exl3frac2's generated qtf2_frac_gen.cu (git-ignored; written by exl3frac2.load from
the fork's quant/frac.cu before the stamp is taken, so every GPU run has it). Prints JSON."""
import fnmatch, hashlib, json, os, subprocess, sys

RXM = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True).stdout


def ls(repo, rev, path):
    return [x for x in git(repo, "ls-tree", "-r", "--name-only", rev, "--", path).decode().split("\n") if x]


def blob(repo, rev, p):
    return git(repo, "show", f"{rev}:{p}")


def frac_gen(fork, rev):
    """exl3frac2.load's generated copy of the fork's frac.cu"""
    fs = blob(fork, rev, "exllamav3/exllamav3_ext/quant/frac.cu").decode()
    a, b = "KA >= 1 && KA <= 7", "KA >= 0 && KA <= 7"
    assert fs.count(a) == 1
    return fs.replace(a, b).replace('#include "../util.h"', '#include "util.h"').replace('#include "../util.cuh"', '#include "util.cuh"').encode()


def main():
    rev, fork, frev = sys.argv[1:4]; standin = "--standin" in sys.argv
    items = []                                            # (basename, bytes) in engine.code_stamp's order
    rxq = sorted(p for p in ls(RXM, rev, "pipeline/rxq") if fnmatch.fnmatch(p, "pipeline/rxq/*.py") and p.count("/") == 2)
    items += [(os.path.basename(p), blob(RXM, rev, p)) for p in rxq]
    items.append(("quantize.py", blob(fork, frev, "exllamav3/modules/quant/exl3_lib/quantize.py")))
    src = "exllamav3/exllamav3_ext"
    ext = sorted(p for p in ls(fork, frev, src) if p.rsplit(".", 1)[-1] in ("cu", "cuh", "cpp", "h", "py"))
    items += [(os.path.basename(p), blob(fork, frev, p)) for p in ext]
    for m in ("exl3frac", "exl3cb", "exl3frac2"):
        d = f"v3/e3/{m}"
        fs = {p: blob(RXM, rev, p) for p in ls(RXM, rev, d) if p.count("/") == 3 and "." in os.path.basename(p)
              and not os.path.basename(p).startswith(".")}
        if m == "exl3frac2": fs[f"{d}/qtf2_frac_gen.cu"] = frac_gen(fork, frev)
        items += [(os.path.basename(p), fs[p]) for p in sorted(fs)]
    h = hashlib.sha256()
    for name, b in items: h.update(name.encode()); h.update(b)
    if standin: h.update(b"cpu-standin")
    sys.path.insert(0, os.path.join(RXM, "pipeline"))
    import tomllib
    from rxq import config as C
    cfg = tomllib.loads(blob(RXM, rev, "pipeline/config/v3.toml").decode())
    print(json.dumps({"rxm": git(RXM, "rev-parse", "--short", rev).decode().strip(), "fork": git(fork, "rev-parse", "--short", frev).decode().strip(),
                      "settings_stamp": C.settings_stamp(cfg), "code_stamp": h.hexdigest()[:16], "kernels": "cpu-standin" if standin else "gpu",
                      "files": len(items), "rxq_py": len(rxq), "ext_files": len(ext)}))


if __name__ == "__main__":
    main()
