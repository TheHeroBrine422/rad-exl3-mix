# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Small helpers shared by the server, the worker and the checker (stdlib only)."""
import hashlib, json, os, re, struct, tempfile, time

API = "v0"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")          # one path component
RELPATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}(/[A-Za-z0-9][A-Za-z0-9._@-]{0,159}){0,7}$")


def now():
    return time.time()


def canon(obj):
    """canonical JSON (the input of every stamp)"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def stamp(obj, n=16):
    return hashlib.sha256(canon(obj).encode()).hexdigest()[:n]


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()


_UMASK = os.umask(0o022); os.umask(_UMASK)                             # read once, at import (single-threaded)


def write_json_atomic(path, obj):
    """write via a temp file + rename; the file gets the mode open() would give it (mkstemp's is 0600, which hid
    manifest.json from a server running under another uid, e.g. the container)"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o666 & ~_UMASK)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except FileNotFoundError:
        return default


def safe_relpath(p):
    """a manifest path: relative, no '..', no empty or dot-led components"""
    return isinstance(p, str) and bool(RELPATH_RE.match(p)) and ".." not in p.split("/")


def st_header(path):
    """safetensors header (dict) without reading the data; raises ValueError on a malformed file"""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        b = f.read(8)
        if len(b) != 8: raise ValueError("short file")
        n = struct.unpack("<Q", b)[0]
        if n > min(size - 8, 100 << 20): raise ValueError("bad header length")
        hdr = json.loads(f.read(n))
    if not isinstance(hdr, dict): raise ValueError("header is not an object")
    end = 0
    for k, v in hdr.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]
        if not (0 <= a <= b): raise ValueError(f"{k}: bad offsets")
        end = max(end, b)
    if 8 + n + end != size: raise ValueError(f"size {size} != header end {8 + n + end}")
    return hdr


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB": return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024
