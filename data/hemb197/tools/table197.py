#!/usr/bin/env python3
"""table197.py SCORES_DIR GOLDEN_DIR OUT_DIR -- #197 (w185): the two headline tables from the per-arm reads.
Inputs: SCORES_DIR/<arm>.json (radiance's summary) and <arm>.json.rows (float32, n_rows x k: column 0 = KL of each golden
row), <arm>.vram.tsv (sampler, score197x only); GOLDEN_DIR = the t183 golden (kld.json: docs with row0/tokens/score_from;
gen.u8 mask). Per arm: mean KL over gen rows (kl_mean.py's definition) and the paired per-document bootstrap of the
difference to the base's arm (a) (paired.py's: 2000 resamples of documents, seed 0). Writes OUT_DIR/arms.json (raw
numbers), OUT_DIR/table.md (the two tables). Sizes (bytes) come from the menu (ours) and the competitor's container listing
(rad-info, parsed); the VRAM columns from the engine's startup report (settled free) and the sampler when present."""
import glob, json, math, os, re, sys
import numpy as np

SD, GD, OD = sys.argv[1:4]
os.makedirs(OD, exist_ok=True)
kld = json.load(open(f"{GD}/kld.json"))
n = kld["rows"]
gen = np.fromfile(f"{GD}/gen.u8", np.uint8).astype(bool)
assert gen.size == n
docs = kld["docs"]
P = 26895998464
FIXED = 35668992
MENU = {  # bytes of our options (menu173m-v2.json, the same numbers the search uses)
    "head": {"u5.375": 854976000, "u3.75": 596723200, "u2.75": 437798400, "bf16": 2542796800},
    "emb": {"t4.625": 735782400, "t3.375": 537126400, "t1.875": 298739200, "bf16": 2542796800},
}
TOTAL = {"5.5": 18490937344, "4.0": 13447983104, "2.5": 8404987904}


def read_rows(name):
    a = np.fromfile(f"{SD}/{name}.json.rows", np.float32)
    k = a.size // n
    return a.reshape(n, k)[:, 0].astype(np.float64)


def paired(kl_a, kl_b):
    """paired.py's statistic on column-0 KL vectors: means over gen rows, the difference and its doc-bootstrap SE."""
    sa, sb, ns = [], [], []
    for d in docs:
        kk = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + kk); mk = gen[s]
        sa.append(kl_a[s][mk].sum()); sb.append(kl_b[s][mk].sum()); ns.append(mk.sum())
    sa, sb, ns = map(np.array, (sa, sb, ns))
    ka, kb = sa.sum() / ns.sum(), sb.sum() / ns.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(ns), len(ns)); bs.append((sb[i].sum() - sa[i].sum()) / ns[i].sum())
    return ka, kb, float(np.std(bs))


def sampler_peak(name):
    f = f"{SD}/{name}.vram.tsv"
    if not os.path.exists(f): return None
    rows = [l.split() for l in open(f) if l.strip()]
    by = {}
    for t, card, used, total in rows:
        by.setdefault(card, []).append(int(used))
    best = None
    for card, v in by.items():
        if len(v) < 3: continue
        delta = max(v) - v[0]                       # rise over the first sample, before the read's allocations
        if best is None or delta > best[1]: best = (card, delta, max(v))
    return best


def settled_free(name):
    f = f"{SD}/../logs/{name}.log"
    if not os.path.exists(f): return None
    m = re.findall(r"card \d+ settled at ([\d.]+) GiB free", open(f, errors="replace").read())
    return float(m[-1]) if m else None


arms = {}
for f in sorted(glob.glob(f"{SD}/*.json.rows")):
    name = os.path.basename(f)[:-len(".json.rows")]
    try:
        kl = read_rows(name)
    except Exception as e:
        print("skip", name, e); continue
    arms[name] = {"kl_mean_gen": float(kl[gen].mean()), "kl": kl, "rows_bytes": os.path.getsize(f),
                  "sampler_peak": sampler_peak(name), "settled_free_GiB": settled_free(name)}

ref = {"5.5": "5.5-a", "4.0": "4.0-a", "2.5": "2.5-a"}
out = {}
for name, a in arms.items():
    rec = {k: v for k, v in a.items() if k != "kl"}
    base = name.split("-")[0]
    if base in ref and ref[base] in arms and name != ref[base] and name.split("-")[1] in "bcde":
        ka, kb, se = paired(arms[ref[base]]["kl"], a["kl"])
        rec["vs_a"] = {"kld_a": ka, "rel_pct": 100 * (kb - ka) / ka, "diff": kb - ka, "se": se, "z": (kb - ka) / se}
    per = []
    for d in docs:
        kk = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + kk); mk = gen[sl]
        per.append([float(a["kl"][sl][mk].sum()), int(mk.sum())])
    rec["perdoc_sum_n"] = per                       # per document: [sum of KL over gen rows, gen rows] (reproduces the CIs)
    out[name] = rec
json.dump(out, open(f"{OD}/arms.json", "w"), indent=1)

L = []
L.append("## Q1: KL-IQR's head / embedding in our trunk (t183, whole slice, gen rows, bf16 KV)\n")
THEIRS = {"small": {"head": 797604085, "emb": 804561879}, "tiny": {"head": 638681907, "emb": 486707036}}
OURS = {"5.5": {"head": ("u5.375", 854976000), "emb": ("t4.625", 735782400), "theirs": "small"},
        "4.0": {"head": ("u3.75", 596723200), "emb": ("t3.375", 537126400), "theirs": "tiny"},
        "2.5": {"head": ("u2.75", 437798400), "emb": ("t1.875", 298739200), "theirs": "tiny"}}
L.append("Bytes at known cost (container bytes of the head and the embedding):\n")
L.append("| base | ours head | ours head bytes | their head | their head bytes | ours emb | ours emb bytes | their emb | their emb bytes |")
L.append("|---|---|---|---|---|---|---|---|---|")
for base in ("5.5", "4.0", "2.5"):
    o = OURS[base]; t = THEIRS[o["theirs"]]
    L.append(f"| {base} | {o['head'][0]} | {o['head'][1]:,} | {o['theirs']} | {t['head']:,} | {o['emb'][0]} | {o['emb'][1]:,} | {o['theirs']} | {t['emb']:,} |")
L.append("")
L.append("| base | arm | head in the serve | embedding in the serve | t183 KLD | rel vs (a) | 95 % CI on rel |")
L.append("|---|---|---|---|---|---|---|")
DESC = {"a": ("ours", "ours"), "b": ("true bf16", "ours"), "c": ("their dequantized (bf16)", "ours"),
        "d": ("ours", "true bf16"), "e": ("ours", "their dequantized (bf16)")}
for base in ("5.5", "4.0", "2.5"):
    o = OURS[base]
    for arm in "abcde":
        nm = f"{base}-{arm}"
        if nm not in out: 
            L.append(f"| {base} | {arm} | {DESC[arm][0] if DESC[arm][0] != 'ours' else o['head'][0]} | {DESC[arm][1] if DESC[arm][1] != 'ours' else o['emb'][0]} | (not run) | | |"); continue
        r = out[nm]
        hd = DESC[arm][0] if DESC[arm][0] != "ours" else o["head"][0]
        ed = DESC[arm][1] if DESC[arm][1] != "ours" else o["emb"][0]
        if arm == "c": hd = f"their {o['theirs']} head (bf16 donor)"
        if arm == "e": ed = f"their {o['theirs']} emb (bf16 donor)"
        vs = r.get("vs_a")
        if vs:
            rel = f"{vs['rel_pct']:+.2f} %"; ci = f"±{100 * 1.96 * vs['se'] / vs['kld_a']:.2f} %"
        else:
            rel, ci = "—", "—"
        L.append(f"| {base} | {arm} | {hd} | {ed} | {r['kl_mean_gen']:.6f} | {rel} | {ci} |")
L.append("")
L.append("Bytes of the bf16 donors are 2,542,796,800 each (plain bf16, the core's own encoding); the base's own bf16 head / embedding are byte-identical to the checkpoint (rel_fro 0). Their head = the sum of output.kB weights, ids and vrow.\n")
L.append("## Q2: bf16 embedding off the card (host RAM), quality and VRAM\n")
CARD_GIB = 34208743424 / 2**30          # one R9700 (sysfs mem_info_vram_total)
def used(name):
    f = settled_free(name)
    return None if f is None else CARD_GIB - f
L.append("VRAM: the card bytes of each serve (menu accounting: trunk + head + fixed [+ embedding on the card]), and what the engine measured after load (card used = total - the startup report's 'settled free').\n")
L.append("| S | card bytes now (trunk+head+fixed+emb) | card bytes, bf16 emb on host (same trunk) | engine used now (GiB) | engine used, bf16 emb on host (GiB) | t183 now (a) | t183 bf16-emb at S (d) | S' (label, bf16-emb on host) | t183 at S' (same card bytes as now) |")
L.append("|---|---|---|---|---|---|---|---|---|")
SPRIME = {"2.5": ("sp3.26", "3.2563"), "4.0": ("sp4.76", "4.7563"), "5.5": ("sp6.26", "6.2563")}
for base in ("2.5", "4.0", "5.5"):
    a_ = out.get(f"{base}-a"); d_ = out.get(f"{base}-d"); dram = out.get(f"{base}-d-ram"); vs = out.get(f"{base}-a-vs")
    sp = out.get(SPRIME[base][0])
    now_b = TOTAL[base]; host_b = now_b - OURS[base]["emb"][1]
    fmt = lambda x: "—" if x is None else f"{x['kl_mean_gen']:.6f}"
    u_now = used(f"{base}-a-vs"); u_host = used(f"{base}-d-ram")
    sp_txt = "—" if sp is None else f"{sp['kl_mean_gen']:.6f} ({100 * (sp['kl_mean_gen'] - a_['kl_mean_gen']) / a_['kl_mean_gen']:+.2f} % vs a)" if a_ else f"{sp['kl_mean_gen']:.6f}"
    L.append(f"| {base} | {now_b:,} ({now_b * 8 / P:.4f} bpw) | {host_b:,} ({host_b * 8 / P:.4f} bpw) | {'—' if u_now is None else f'{u_now:.2f}'} | {'—' if u_host is None else f'{u_host:.2f}'} | {fmt(a_)} | {fmt(d_)} | {SPRIME[base][1]} (card {now_b * 8 / P:.4f} bpw) | {sp_txt} |")
L.append("")
L.append("S' = the size at which the trunk + head bytes equal ours at S plus the embedding's bytes (card bytes of S' = total bytes of S); its probe carries the bf16 embedding on the host, so its container label is higher (the bf16 table adds 2,542,796,800 B).\n")
open(f"{OD}/table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
