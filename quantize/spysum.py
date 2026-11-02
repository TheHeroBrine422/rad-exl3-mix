#!/usr/bin/env python3
"""spysum.py SPY.txt [N] -- summary of a py-spy raw (collapsed-stack) profile of one ourq run (record --native --idle,
main thread and helpers): samples by leaf class (GPU wait / HIP runtime / torch CPU / Python / I/O / idle), the top N
leaf frames, and inclusive time of the quantize-module and ourq functions on the stacks."""
import collections, re, sys
f, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25
leaf, incl, cls, tot = collections.Counter(), collections.Counter(), collections.Counter(), 0
def classify(fr):
    s = fr.lower()
    if any(k in s for k in ("hipstreamsynchronize", "hipdevicesynchronize", "hipeventsynchronize", "synchronize", "wait", "futex", "sched_yield", "hsa_signal")): return "GPU / thread wait"
    if any(k in s for k in ("hip", "hsa", "amdhip", "rocr", "libhsa", "kfd", "ioctl")): return "HIP runtime / launch"
    if any(k in s for k in ("read", "mmap", "safe_open", "get_tensor", "copy_user", "memcpy", "pread")): return "I/O / copies"
    if any(k in s for k in ("libtorch", "at::", "c10::", "torch::", "aten")): return "torch CPU (dispatch, ops)"
    if ".py" in s: return "Python"
    return "other native"
for line in open(f):
    line = line.rstrip()
    if not line: continue
    stack, _, n = line.rpartition(" ")
    n = int(n); tot += n
    frames = stack.split(";")
    leaf[frames[-1][:110]] += n
    cls[classify(frames[-1])] += n
    seen = set()
    for fr in frames:
        m = re.match(r"(\w+) \((?:[^)]*/)?(quantize\.py|ourq\.py|qprof3\.py)", fr.strip())
        if m and m.group(1) not in seen: seen.add(m.group(1)); incl[f"{m.group(2)}:{m.group(1)}"] += n
print(f"{tot} samples")
print("by leaf class:"); [print(f"  {c:28s} {v / tot:6.1%}") for c, v in cls.most_common()]
print(f"top {N} leaf frames:"); [print(f"  {v / tot:6.1%}  {k}") for k, v in leaf.most_common(N)]
print("inclusive (quantize.py / ourq.py functions):"); [print(f"  {v / tot:6.1%}  {k}") for k, v in incl.most_common(N)]
