"""w162 (#185 profile): disk read floor for a set of files, O_DIRECT (page cache bypassed), page-aligned buffer.

    python -I floor_read.py FILE [FILE ...]      -> one line: bytes, seconds, GB/s

dd iflag=direct fails on files whose size is not a multiple of the block size (EINVAL on the last read), so this
reads with preadv into an aligned mmap buffer and treats the short tail read as the end."""
import mmap, os, sys, time

BUF = 16 << 20


def main(paths):
    buf = mmap.mmap(-1, BUF)                       # anonymous mapping: page-aligned
    total = 0; t0 = time.time()
    for p in paths:
        fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
        try:
            off = 0
            while True:
                n = os.preadv(fd, [buf], off)
                if n <= 0:
                    break
                total += n; off += n
                if n < BUF:
                    break
        finally:
            os.close(fd)
    dt = time.time() - t0
    print(f"floor read {total} bytes direct in {dt:.2f} s = {total / dt / 1e9:.2f} GB/s over {len(paths)} files")


if __name__ == "__main__":
    main(sys.argv[1:])
