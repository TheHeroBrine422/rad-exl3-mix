#!/usr/bin/env python3
"""plan.py OUTDIR -- w43 job lists (README.md): every job of items 1, 2, 3, 1b, 1c in priority order, split between
cpu2 (Thor) and cpu1 (colo) by a list-scheduling simulation with estimated costs; writes OUTDIR/jobs-cpu1.txt,
jobs-cpu2.txt (lines "ITEM MODE LAYER TENSOR K SEED ROWS") and prints the simulated finish time of each item.
Cost model (Thor core-seconds, measured 2026-11-06 smoke runs): tiles / 88 per second + 50 s scale search + load;
colo cores run at SPEED_COLO of that (its smoke run, beside w38's jobs: ~0.6)."""
import heapq, os, sys

T = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
     "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
     "ffn_down": (2, 30, 63)}
KD = {"ssm_out": 6144, "attn_output": 6144, "ffn_down": 17408}
REP = {"gain": 13, "damp": 9, "cd": 2.5}            # stacked LDLQ-pass equivalents per job
ROWS = 128
SLOTS = {"cpu2": (30, 5), "cpu1": (16, 2)}          # (jobs, of which ffn_down; Thor: 3 beside #94, ~7 after)
SPEED = {"cpu2": 1.0, "cpu1": float(os.environ.get("SPEED_COLO", 0.6))}


def cost(mode, w):
    kd = KD.get(w, 5120)
    return ROWS * REP[mode] / 16 * kd / 16 / 88 + 50 + (60 if w == "ffn_down" else 15)


def jobs():
    out = []
    def add(item, mode, Ks, seeds):
        js = [(item, mode, l, w, K, s) for s in seeds for w, ls in T.items() for l in ls for K in Ks]
        out.extend(sorted(js, key=lambda j: -cost(j[1], j[3])))        # longest first inside an item
    add("1", "gain", ["1.5", "2", "2.5", "3", "4", "5", "6", "7"], [0, 1])
    add("2", "damp", ["1.5", "2.5", "3.5", "5.5"], [0, 1])
    add("3", "cd", ["1.5", "2", "3", "4", "6"], [0, 1])
    add("1b", "gain", ["1.25", "1.75", "2.25", "2.75"], [0, 1])
    add("1c", "gain", ["1.5", "2", "2.5", "3", "4", "5", "6", "7"], [2])
    return out


def main():
    od = sys.argv[1]
    pin = {}                                                       # jobs already running / done stay on their box
    for h in SLOTS:
        p = os.path.join(od, f"pin-{h}.txt")
        if os.path.exists(p):
            for l in open(p): pin[tuple(l.split()[:4])] = h          # MODE LAYER TENSOR K-SEED key below
    free = {h: [0.0] * n for h, (n, _) in SLOTS.items()}           # slot free times
    down = {h: [0.0] * d for h, (_, d) in SLOTS.items()}           # ffn_down tokens
    lists = {h: [] for h in SLOTS}; fin = {}
    for j in jobs():
        isd = j[3] == "ffn_down"
        best = None
        for h in ([pin[key]] if (key := (j[1], str(j[2]), j[3], f"{j[4]}-{j[5]}")) in pin else SLOTS):
            t = min(free[h])
            if isd: t = max(t, min(down[h]))
            done = t + cost(j[1], j[3]) / SPEED[h]
            if best is None or done < best[0]: best = (done, h, t)
        done, h, t = best
        i = free[h].index(min(free[h])); free[h][i] = done
        if isd: k = down[h].index(min(down[h])); down[h][k] = done
        lists[h].append(j); fin[j[0]] = max(fin.get(j[0], 0), done)
    for h, js in lists.items():
        with open(os.path.join(od, f"jobs-{h}.txt"), "w") as f:
            for j in js: f.write(f"{j[0]} {j[1]} {j[2]} {j[3]} {j[4]} {j[5]} {ROWS}\n")
        cs = sum(cost(j[1], j[3]) for j in js) / 3600
        print(f"{h}: {len(js)} jobs, {cs:.0f} Thor-core-h, {sum(j[3] == 'ffn_down' for j in js)} ffn_down")
    for it, t in fin.items(): print(f"item {it}: simulated finish {t / 3600:.1f} h after start")


if __name__ == "__main__":
    main()
