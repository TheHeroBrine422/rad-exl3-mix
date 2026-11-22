import sys, os, shutil
q, spec = sys.argv[1], sys.argv[2]          # spec: ARM=r1+r2,...;ARM=...
want = {k: [float(x) for x in v.split("+")] for k, v in (x.split("=") for x in spec.split(";"))}
def nm(x):
    t = f"{x:.0e}"; p = 0
    while float(t) != x and p < 6: p += 1; t = f"{x:.{p}e}"
    return "ckpt-raw-" + t
claims = set(os.listdir(os.path.join(q, "claims"))); done = set(os.listdir(os.path.join(q, "done")))
L = open(os.path.join(q, "jobs.txt")).read().split("\n"); out = []; hit = set()
for ln in L:
    t = ln.split(" ")
    if len(t) > 5 and t[0] == "T" and t[1] in want and "--snap-at" not in t:
        key = "T-" + t[1]
        if key in claims or key in done: print("SKIP started", t[1]); out.append(ln); continue
        r = want[t[1]]; t[4] = t[4] + "," + ",".join(nm(x) for x in r)
        i = next((k for k, x in enumerate(t) if x.startswith("@")), len(t))
        t = t[:i] + ["--snap-at", ",".join(f"{x:g}" for x in r)] + t[i:]; ln = " ".join(t); hit.add(t[1])
    out.append(ln)
print("edited", sorted(hit), "missing", sorted(set(want) - hit))
shutil.copy(os.path.join(q, "jobs.txt"), os.path.join(q, "jobs.txt.pre-snap"))
open(os.path.join(q, "jobs.txt.new"), "w").write("\n".join(out)); os.replace(os.path.join(q, "jobs.txt.new"), os.path.join(q, "jobs.txt"))
