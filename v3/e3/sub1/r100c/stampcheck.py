#!/usr/bin/env python3
"""stampcheck.py ORIG_OURQ PATCHED_OURQ -- after applying ourq-hash.patch: the code stamp hashes ourq.py through
_io_canon (+io lines dropped, #io-was: lines restored); the patched file's canon (its own _io_canon: + the +cb / #cb-was:
pair) must equal the original's byte for byte, so every existing codebook keeps its stamp. Exit 1 if not."""
import sys
old = lambda b: b"".join(x[8:] if x.startswith(b"#io-was:") else x for x in b.splitlines(True) if not x.rstrip().endswith(b"# +io"))
new = lambda b: b"".join(x[8:] if x.startswith((b"#io-was:", b"#cb-was:")) else x for x in b.splitlines(True)
                         if not x.rstrip().endswith((b"# +io", b"# +cb")))
o, p = (open(f, "rb").read() for f in sys.argv[1:3])
ok = old(o) == new(p)
print("canon identical (stamps unchanged for mul1 / i8 / i4 / i8c):", ok)
sys.exit(0 if ok else 1)
