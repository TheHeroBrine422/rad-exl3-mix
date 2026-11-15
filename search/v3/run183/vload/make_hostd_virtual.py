#!/usr/bin/env python3
"""make_hostd_virtual.py HOSTD_IN HOSTD_OUT -- writes pipeline/rxrun/hostd.py with the "virtual" eval option (w162, #185):
a candidate is served as a manifest (`base BASE` + the mixfile) that radiance reads in place from its donors; no mix .rad is
written, so no radmix step. The default (no eval.virtual in HOSTCONF) is the old path, byte for byte. The patch is
`diff -u HOSTD_IN HOSTD_OUT` (hostd-virtual.patch). Each edit must match exactly once, or this exits 1."""
import sys

EDITS = []

EDITS.append((
    '''Census (w142, #178; HOSTCONF eval.census_cmd, optional): run on every mix before it is scored ({mix} {mixfile} {base}, e.g.
store/census.py {mix} --mix {mixfile} --base {base} --fast [--expect-act a8g128 ...]); a non-zero exit fails the unit
unscored. A pass is remembered in <mix>.census (newer than the mix: not run again).
''',
    '''Census (w142, #178; HOSTCONF eval.census_cmd, optional): run on every mix before it is scored ({mix} {mixfile} {base}, e.g.
store/census.py {mix} --mix {mixfile} --base {base} --fast [--expect-act a8g128 ...]); a non-zero exit fails the unit
unscored. A pass is remembered in <mix>.census (newer than the mix: not run again).
Virtual candidates (w162, #185; HOSTCONF eval.virtual = true): no mix file. The unit writes the manifest <tag>.mix (a
`base BASE` line, then the mixfile lines, which go to <tag>.mixfile), radiance reads it in place (rad-convert's packs are
the donors), and the scorer gets the manifest as its model. eval.virtual_bytes_cmd (e.g. [RAD_INFO, "--bytes", "{mix}"])
gives the served size; eval.census_virtual_cmd (optional; search/v3/run183/vload/census_manifest.py with
{mix} {mixfile} {base}) runs in place of census_cmd.
''',
))

EDITS.append((
    '''    mf = f"{MX}/{tag}.mix"; mrad = f"{MX}/{tag}.rad"
''',
    '''    virt = bool(E.get("virtual"))                                 # w162 (#185): a manifest, not a mix file
    mf = f"{MX}/{tag}.mixfile" if virt else f"{MX}/{tag}.mix"
    mrad = f"{MX}/{tag}.mix" if virt else f"{MX}/{tag}.rad"
''',
))

EDITS.append((
    '''    open(mf, "w").write("\\n".join(lines) + "\\n")
    if not os.path.exists(mrad):
''',
    '''    open(mf, "w").write("\\n".join(lines) + "\\n")
    if virt:
        if not E.get("virtual_bytes_cmd"): return {"rc": 1, "err": "virtual: HOSTCONF eval.virtual_bytes_cmd (rad-info --bytes) is needed"}
        open(mrad, "w").write(f"base {rp(mx['base'])}\\n" + open(mf).read())
    elif not os.path.exists(mrad):
''',
))

EDITS.append((
    '''    if E.get("census_cmd"):                                       # w142 (#178, mining B2): the container must be what the mix
        ok = mrad + ".census"                                     # says before a KLD means anything (store/census.py)
        if not (os.path.exists(ok) and os.path.getmtime(ok) >= os.path.getmtime(mrad)):
            rc = sh(fill(E["census_cmd"], mix=mrad, mixfile=mf, base=rp(mx["base"])), log)
            if rc: return {"rc": 1, "err": f"census {tag} rc {rc}: the container differs from its mix / expectations ({log}); not scored"}
            open(ok, "w").write("CENSUS PASS\\n")
    import numpy as np
    nbytes = os.path.getsize(mrad); res = {"rc": 0, "kind": "eval", "tag": tag, "host": H.name, "bytes": nbytes,
''',
    '''    cen = E.get("census_virtual_cmd") if virt else E.get("census_cmd")
    if cen:                                                       # w142 (#178, mining B2): the container must be what the mix
        ok = mrad + ".census"                                     # says before a KLD means anything (store/census.py)
        if virt or not (os.path.exists(ok) and os.path.getmtime(ok) >= os.path.getmtime(mrad)):   # a manifest is rewritten each time
            rc = sh(fill(cen, mix=mrad, mixfile=mf, base=rp(mx["base"])), log)
            if rc: return {"rc": 1, "err": f"census {tag} rc {rc}: the container differs from its mix / expectations ({log}); not scored"}
            open(ok, "w").write("CENSUS PASS\\n")
    import numpy as np
    if virt:                                                      # the served size, from radiance (rad-info --bytes)
        p = subprocess.run(fill(E["virtual_bytes_cmd"], mix=mrad), stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if p.returncode or not p.stdout.strip().isdigit():
            return {"rc": 1, "err": f"virtual bytes {tag}: rc {p.returncode} {p.stderr.strip()[-200:]}"}
        nbytes = int(p.stdout)
    else:
        nbytes = os.path.getsize(mrad)
    res = {"rc": 0, "kind": "eval", "tag": tag, "host": H.name, "bytes": nbytes,
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
