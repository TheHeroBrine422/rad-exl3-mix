#!/usr/bin/env python3
"""scan.py INDEX.sqlite PATH... -- the tensor store's index: every weight entry of every .rad pack under PATH.

A *pack* is any RAD2 container (existing level containers, releases, small experiment containers); nothing is
rewritten, so radmix keeps cloning weights out of packs by reference. A *variant* is one encoded tensor, identified
by content: vid = sha256(name | shape | the encoding struct | each plane's bytes) -- identical tensors in several
packs (e.g. one of turboderp's tensors carried by several of his size containers, or a release's reflinked copy)
are ONE variant with several locations. Incremental: a pack already indexed with the same size and mtime is skipped.

Tables:
  packs(path, bytes, mtime, arch, recipe, created_by, n_entries)
  variants(vid, name, shape, scheme, quantizer, options, bytes)       -- one row per distinct tensor
  locs(vid, path, offset)                                            -- where each variant lives
Report: store/report.py. Env SCAN_SKIP (comma-separated path substrings to skip)."""
import glob, hashlib, json, os, sqlite3, struct, sys, time

HDR = struct.Struct("<IIQ5Q" + "Q" * 18)      # magic, ver, file_bytes, 5 stroffs, 9 (off, count) pairs
ENT = struct.Struct("<QQQII6qiiiIQQQ2Q")       # RadFileEntry, 136 bytes
ENC_SIZE = 24 + 24 + 4 + 4 + 8 * 64            # RadEncoding (RAD_ENC_STR 24, 8 planes of 64 bytes)
assert ENT.size == 136
MAGIC, VER = 0x31444152, 2
CH = 1 << 24


def cstr(blob, off):
    return "" if off == 0 else blob[off:blob.index(b"\0", off)].decode()


def scan_pack(db, path):
    st = os.stat(path)
    row = db.execute("SELECT bytes, mtime FROM packs WHERE path = ?", (path,)).fetchone()
    if row and row[0] == st.st_size and row[1] == int(st.st_mtime): return 0, 0
    with open(path, "rb") as f:
        h = HDR.unpack(f.read(HDR.size))
        if h[0] != MAGIC or h[1] != VER: print(f"skip {path}: not RAD2", flush=True); return 0, 0
        stroffs = h[3:8]
        (str_off, str_n, _, _, dir_off, dir_n, pl_off, pl_n, enc_off, enc_n) = h[8:18]
        f.seek(str_off); S = f.read(str_n)
        f.seek(dir_off); D = f.read(dir_n * ENT.size)
        f.seek(pl_off); P = f.read(pl_n * 16)
        f.seek(enc_off); E = f.read(enc_n * ENC_SIZE)
        arch, _, _, recipe, created = (cstr(S, o) for o in stroffs)
        db.execute("DELETE FROM locs WHERE path = ?", (path,))
        nb = 0
        for i in range(dir_n):
            e = ENT.unpack_from(D, i * ENT.size)
            name, quant, opts = cstr(S, e[0]), cstr(S, e[1]), cstr(S, e[2])
            enc, rank, shape = e[3], e[4], list(e[5:5 + e[4]])
            n_planes, plane_first, offset, nbytes = e[14], e[15], e[16], e[17]
            eb = E[enc * ENC_SIZE:(enc + 1) * ENC_SIZE]
            hs = hashlib.sha256(json.dumps([name, shape]).encode()); hs.update(eb)
            for p in range(n_planes):
                po, pn = struct.unpack_from("<QQ", P, (plane_first + p) * 16)
                hs.update(struct.pack("<QQ", po - offset, pn))
                f.seek(po); left = pn
                while left:
                    b = f.read(min(CH, left)); hs.update(b); left -= len(b)
                nb += pn
            vid = hs.hexdigest()[:32]
            scheme = eb[:24].split(b"\0")[0].decode()
            db.execute("INSERT OR IGNORE INTO variants VALUES (?,?,?,?,?,?,?)",
                       (vid, name, json.dumps(shape), scheme, quant, opts, nbytes))
            db.execute("INSERT INTO locs VALUES (?,?,?)", (vid, path, offset))
    db.execute("INSERT OR REPLACE INTO packs VALUES (?,?,?,?,?,?,?)",
               (path, st.st_size, int(st.st_mtime), arch, recipe, created, dir_n))
    db.commit()
    return dir_n, nb


def main():
    db = sqlite3.connect(sys.argv[1])
    db.executescript("""
      CREATE TABLE IF NOT EXISTS packs(path TEXT PRIMARY KEY, bytes INT, mtime INT, arch TEXT, recipe TEXT,
                                       created_by TEXT, n_entries INT);
      CREATE TABLE IF NOT EXISTS variants(vid TEXT PRIMARY KEY, name TEXT, shape TEXT, scheme TEXT, quantizer TEXT,
                                          options TEXT, bytes INT);
      CREATE TABLE IF NOT EXISTS locs(vid TEXT, path TEXT, offset INT);
      CREATE INDEX IF NOT EXISTS locs_vid ON locs(vid); CREATE INDEX IF NOT EXISTS locs_path ON locs(path);
      CREATE INDEX IF NOT EXISTS var_name ON variants(name);""")
    skip = [s for s in os.environ.get("SCAN_SKIP", "").split(",") if s]
    paths = []
    for root in sys.argv[2:]:
        paths += [root] if root.endswith(".rad") else glob.glob(f"{root}/**/*.rad", recursive=True)
    paths = sorted(p for p in set(map(os.path.realpath, paths)) if not any(s in p for s in skip))
    t0 = time.time(); tot = 0
    for i, p in enumerate(paths):
        n, nb = scan_pack(db, p); tot += nb
        if n: print(f"[{i + 1}/{len(paths)}] {p}: {n} entries, {nb / 1e9:.1f} GB hashed, "
                    f"{tot / 1e9 / max(time.time() - t0, 1e-9):.2f} GB/s", flush=True)
    # stale packs (deleted files)
    for (p,) in db.execute("SELECT path FROM packs").fetchall():
        if not os.path.exists(p):
            db.execute("DELETE FROM locs WHERE path = ?", (p,)); db.execute("DELETE FROM packs WHERE path = ?", (p,))
    db.execute("DELETE FROM variants WHERE vid NOT IN (SELECT vid FROM locs)"); db.commit()
    print(f"SCAN DONE {len(paths)} packs, {tot / 1e9:.1f} GB hashed in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
