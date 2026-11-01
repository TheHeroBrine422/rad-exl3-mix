#!/usr/bin/env python3
"""sizedef.py BOARD_BYTES:CONTAINER.rad ... -- mining merge B10: what does the leaderboard's `bytes` count?
Per container: file size, sum of entry nbytes, sum of plane bytes, bytes per tensor group (embedding, head, trunk,
norms / other), and the gap of each candidate definition to the board's number."""
import os, struct, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan import HDR, ENT, cstr, MAGIC, VER
for arg in sys.argv[1:]:
    board, path = arg.split(":", 1); board = int(board)
    with open(path, "rb") as f:
        h = HDR.unpack(f.read(HDR.size)); assert h[0] == MAGIC and h[1] == VER
        (str_off, str_n, _, _, dir_off, dir_n, pl_off, pl_n, enc_off, enc_n) = h[8:18]
        f.seek(str_off); S = f.read(str_n); f.seek(dir_off); D = f.read(dir_n * ENT.size)
        f.seek(pl_off); P = f.read(pl_n * 16)
    ent, planes, grp = 0, 0, collections.Counter()
    for i in range(dir_n):
        e = ENT.unpack_from(D, i * ENT.size); name = cstr(S, e[0])
        pb = sum(struct.unpack_from("<QQ", P, (e[15] + p) * 16)[1] for p in range(e[14]))
        ent += e[17]; planes += pb
        g = ("embed" if "embd" in name or "embed" in name else "head" if name.startswith("output.") else
             "mtp" if "mtp" in name else "vision" if name.startswith(("v.", "mm.", "visual")) else
             "trunk" if name.endswith(".weight") and e[4] == 2 else "small")
        grp[g] += pb
    fs = os.path.getsize(path)
    print(f"{os.path.basename(path)}: board {board}  file {fs} ({fs - board:+d})  entries {ent} ({ent - board:+d})  "
          f"planes {planes} ({planes - board:+d})")
    print("   groups: " + ", ".join(f"{k} {v}" for k, v in grp.most_common()) +
          f"   planes-small {planes - grp['small']} ({planes - grp['small'] - board:+d})")
