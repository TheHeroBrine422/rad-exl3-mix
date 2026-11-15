#!/usr/bin/env python3
"""make_hostd_final.py HOSTD_IN HOSTD_OUT -- writes pipeline/rxrun/hostd.py with the "final" eval option (w162, #185 B) on
top of the "virtual" one (make_hostd_virtual.py's edits, applied first): with HOSTCONF eval.final_root = DIR, a candidate's
trunk lines are `<weight> final DIR/<level>/final codebook=CB` (CB = eval.final_codebook, default mad), the levels' final/
directories on this host. There is no pack step (no level packs are built or kept), and the manifest is served as the
virtual one (virtual_bytes_cmd, census_virtual_cmd). Without eval.final_root the code path is the virtual one, and without
eval.virtual either, the old one. The patch is `diff -u HOSTD_IN HOSTD_OUT` (hostd-final.patch). Each edit must match
exactly once, or this exits 1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_hostd_virtual import EDITS as VIRTUAL_EDITS  # noqa: E402

EDITS = list(VIRTUAL_EDITS)

EDITS.append((
    '''gives the served size; eval.census_virtual_cmd (optional; search/v3/run183/vload/census_manifest.py with
{mix} {mixfile} {base}) runs in place of census_cmd.
''',
    '''gives the served size; eval.census_virtual_cmd (optional; search/v3/run183/vload/census_manifest.py with
{mix} {mixfile} {base}) runs in place of census_cmd.
Final candidates (w162, #185 B; HOSTCONF eval.final_root = DIR, eval.final_codebook = mad by default): the trunk lines are
`<weight> final DIR/<level>/final codebook=CB` (the levels' final/ directories on this host), so no level pack is built or
kept; the manifest is served as a virtual one. census_manifest.py refuses a final line: the census of such a candidate is
rad-info --weights-dump of it against the pack (search/v3/run183/vload/cmp_dumps.py).
''',
))

EDITS.append((
    '''    need = sorted(set(lv.values()))
    miss = [n for n in need if not H.current(n)]
    if miss: return {"rc": 1, "err": f"levels not built on this host: {miss}"}
''',
    '''    need = sorted(set(lv.values()))
    fr = E.get("final_root")                                      # w162 (#185, B): trunk lines from the levels' final/ dirs
    if fr:
        miss = [n for n in need if not os.path.exists(f"{fr}/{n}/final/model.safetensors.index.json")]
        if miss: return {"rc": 1, "err": f"final/ not on this host under {fr}: {miss}"}
    else:
        miss = [n for n in need if not H.current(n)]
        if miss: return {"rc": 1, "err": f"levels not built on this host: {miss}"}
''',
))

EDITS.append((
    '''    for n in need:''',
    '''    for n in ([] if fr else need):''',
))

EDITS.append((
    '''virt = bool(E.get("virtual"))''',
    '''virt = bool(E.get("virtual")) or bool(fr)''',
))

EDITS.append((
    '''    lines = [f"{w} {pkn(lv[w])}" for w in trunk] + [f"{w} {rp(p)}" for w, p in (mx.get("donors") or {}).items()]
''',
    '''    cb = E.get("final_codebook", "mad")                           # w162 (#185, B): the codebook the levels' stamps record
    trunk_lines = [f"{w} final {fr}/{lv[w]}/final codebook={cb}" for w in trunk] if fr else [f"{w} {pkn(lv[w])}" for w in trunk]
    lines = trunk_lines + [f"{w} {rp(p)}" for w, p in (mx.get("donors") or {}).items()]
''',
))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"edit does not match once (found {n}): {old[:70]!r}")
        text = text.replace(old, new)
    open(dst, "w").write(text)
    print("wrote", dst)


if __name__ == "__main__":
    main()
