#!/usr/bin/env python3
"""check_mix.py BACKUP_SWAP27 -- S8 text check (CPU, instant): rxq.mix.recipe / level_mix against the swap27 arms'
recipe and mix files as swap27.sh / swap27k.sh wrote them on gpubox (the backup copy of /data/swap27: *.recipe, *.mix;
their level dir and K are read back from the recipe's first line). Byte for byte. Last line "MIX CHECKS n/m PASS"."""
import glob, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxq import mix  # noqa: E402

B = sys.argv[1]; res = []
for rf in sorted(glob.glob(f"{B}/*.recipe")):
    arm = os.path.basename(rf)[:-len(".recipe")]
    txt = open(rf).read()
    m = re.match(r"^\S+ exl3i src=(\S+) codebook=(\S+) rate=(\S+)$", txt.splitlines()[0])
    cb = {v: k for k, v in mix.CB.items()}[m.group(2)]
    r_ok = mix.recipe(m.group(1), float(m.group(3)), cb) == txt
    mf = f"{B}/{arm}.mix"
    m_ok = None
    if os.path.exists(mf):
        mt = open(mf).read(); pack = mt.split()[1]
        m_ok = mix.level_mix(pack) == mt
    ok = r_ok and m_ok is not False
    res.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {arm}: recipe {'identical' if r_ok else 'DIFFERS'}, mix "
          f"{'identical' if m_ok else 'missing' if m_ok is None else 'DIFFERS'} (K{m.group(3)}, {m.group(2)})")
print(f"MIX CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if res and all(res) else 1)
