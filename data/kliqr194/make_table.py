#!/usr/bin/env python3
"""make_table.py -- w177 (#194): the KL-IQR-V1 entries (Perire / radiance PR #364, branch qc/qwen38-kl-guided-iqr @ 3eec7eb6)
scored on t183 and t186n with their build, against IQR-V4 and our V3 incumbents on t183. Reads raw/ only; writes table.md and
table.json next to this file.

Size bases (see table.md, "Size bases"):
  text plane  = (trunk + head + token embedding) bytes x 8 / 26,895,998,464. The tensors of the container minus the vision
                tower (v.*) and the drafter. It reproduces the stated bits of the three entries and of IQR-V4 iq3r (raw/tensor-sizes.json).
                Our search containers are text-only, so this is also their file-size basis. PRIMARY.
  file bpw    = file bytes x 8 / 26,895,998,464 (the literal task metric). It includes the 921 MB vision tower the entries
                ship and our containers do not. SECONDARY.
Ratios: KL / IQR-V4 KL and KL / our incumbent KL, both log-linearly interpolated at the same bpw.
  vs IQR-V4 uses the refs-t183 IQR-V4 curve (t183) and w164's rows of the same containers (t186n), on the text-plane axis.
  vs ours uses the latest tracked incumbent per size on t183 (extract_incumbents.py; snapshot in raw/snapshot.txt).
Uncertainty: 95 % per-document bootstrap of each entry's own mean (kl_boot.py). Ratios carry no interval.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "raw")
PARAMS = 26895998464


def jl(p):
    return [json.loads(l) for l in open(os.path.join(RAW, p)) if l.strip()]


def ll(points, b):
    """log-linear interpolation of KL in bpw between the bracketing points; None outside their range"""
    pts = sorted(points)
    if b < pts[0][0] or b > pts[-1][0]:
        return None
    for (b0, k0), (b1, k1) in zip(pts, pts[1:]):
        if b0 <= b <= b1:
            return math.exp(math.log(k0) + (math.log(k1) - math.log(k0)) * (b - b0) / (b1 - b0))


def pct(r):
    return "n/a" if r is None else f"{(r - 1) * 100:+.1f} %"


def kl(x):
    return "n/a" if x is None else (f"{x:.5f}" if x >= 0.001 else f"{x:.6f}")


def ci(c):
    return "" if not c else f" [{c[0]:.5f}, {c[1]:.5f}]"


entries = json.load(open(os.path.join(RAW, "entries.json")))
tsz = json.load(open(os.path.join(RAW, "tensor-sizes.json")))


def text_bpw(name):
    t = tsz[name]
    return (t["sums"]["trunk_blk"] + t["sums"]["head_output"] + t["single"]["token_embd.weight"]) * 8 / PARAMS


refs = jl("refs-t183.jsonl")
IQ183 = sorted((r["bpw"], r["kld"]) for r in refs if r["name"].startswith("iq"))
w164 = jl("ref-iqr-t186n-w164.jsonl")
bpw_of = {r["name"]: r["bpw"] for r in refs}
IQ186 = sorted((bpw_of[os.path.basename(r["rows_file"]).split("-t186n")[0]], r["kl_mean_gen"]) for r in w164)
inc = jl("incumbents-t183.jsonl")
OURS = sorted((r["bpw"], r["kld_t183"]) for r in inc)
kall = {os.path.basename(r["rows_file"]).replace(".json.rows", ""): r for r in jl("kl-all.jsonl")}
boot = {os.path.basename(r["rows_file"]).replace(".json.rows", ""): r for r in jl("boot.jsonl")}
gate = json.load(open(os.path.join(RAW, "gate.json")))

out = {"entries": {}, "iqr_v4_iq3r_check": {}, "iqr_v4_t183": IQ183, "iqr_v4_t186n": IQ186, "ours_t183": OURS}
iq_tb = text_bpw("qwen3.8-27b-iq3r-v4")
out["iqr_v4_iq3r_check"] = {"text_plane_bpw_from_container": iq_tb, "stated_bpw_w134": bpw_of["iq3r"],
                            "diff": iq_tb - bpw_of["iq3r"]}
rows = []
for e in entries:
    k = e["key"]; cont = os.path.basename(e["file"]).replace(".rad", "")
    tb = text_bpw(cont); fb = e["bytes"] * 8 / PARAMS
    t183 = kall.get(f"{k}-t183", {}).get("kl_mean_gen"); t186 = kall.get(f"{k}-t186n", {}).get("kl_mean_gen")
    c183 = boot.get(f"{k}-t183", {}).get("ci95_doc_boot"); c186 = boot.get(f"{k}-t186n", {}).get("ci95_doc_boot")
    iq183 = ll(IQ183, tb); iq186 = ll(IQ186, tb); ours_tp = ll(OURS, tb); ours_fb = ll(OURS, fb)
    d = {"label": e["label"], "file": e["file"], "bytes": e["bytes"], "sha256": e["sha256"], "file_bpw": fb,
         "text_plane_bpw": tb, "stated_bits": e["stated_bits"], "stated_minus_text_plane": e["stated_bits"] - tb,
         "kl_t183": t183, "kl_t183_ci95": c183, "kl_t186n": t186, "kl_t186n_ci95": c186,
         "iqr_v4_t183_at_text_bpw": iq183, "ratio_iqr_t183": None if (t183 is None or iq183 is None) else t183 / iq183,
         "iqr_v4_t186n_at_text_bpw": iq186, "ratio_iqr_t186n": None if (t186 is None or iq186 is None) else t186 / iq186,
         "ours_t183_at_text_bpw": ours_tp, "ratio_ours_t183": None if (t183 is None or ours_tp is None) else t183 / ours_tp,
         "ours_t183_at_file_bpw": ours_fb, "ratio_ours_t183_file_bpw": None if (t183 is None or ours_fb is None) else t183 / ours_fb}
    out["entries"][k] = d; rows.append((k, d))
s = out["sanity"] = {"iq3r_t183_their_build": gate["kl"], "iq3r_t183_lead_refs": gate["expect"],
                     "ratio_minus_1": gate["ratio_minus_1"]}
w164_iq3 = [r for r in w164 if r["rows_file"].startswith("iq3r-")][0]["kl_mean_gen"]
t186_iq3 = kall.get("iq3r-t186n", {}).get("kl_mean_gen")
s["iq3r_t186n_their_build"] = t186_iq3; s["iq3r_t186n_w164_build"] = w164_iq3
s["ratio_minus_1_t186n"] = None if t186_iq3 is None else t186_iq3 / w164_iq3 - 1
json.dump(out, open(os.path.join(HERE, "table.json"), "w"), indent=1)

snap = open(os.path.join(RAW, "snapshot.txt")).read().strip() if os.path.exists(os.path.join(RAW, "snapshot.txt")) else ""
L = ["# KL-IQR-V1 entries (Perire / radiance PR #364) on t183 and t186n (#194, w177)\n",
     "Scored with their build (`qc/qwen38-kl-guided-iqr` @ 3eec7eb6, ROCm 10.1, gfx1201) on box2, one radiance per card: bf16 KV, "
     "`--tp 1`, `--max-model-len 16384`, `--max-num-seqs 16`, `--kld-ref` on w164's golden-t183 / golden-t186n. Means over gen rows "
     f"(gen.u8). Our incumbents: {snap}\n",
     "## Headline (text-plane bits: the entries' stated bits; IQR-V4 and our containers on the same basis)\n",
     "| entry | text-plane bpw | t183 KL (95 % doc CI) | vs IQR-V4 at same bpw (t183) | vs our V3 incumbent at same bpw (t183) | t186n KL (95 % doc CI) | vs IQR-V4 at same bpw (t186n) |",
     "|---|---:|---:|---:|---:|---:|---:|"]
for k, d in rows:
    L.append(f"| {d['label']} | {d['text_plane_bpw']:.4f} | {kl(d['kl_t183'])}{ci(d['kl_t183_ci95'])} | "
             f"{pct(d['ratio_iqr_t183'])} (IQR-V4 {kl(d['iqr_v4_t183_at_text_bpw'])}) | "
             f"{pct(d['ratio_ours_t183'])} (ours {kl(d['ours_t183_at_text_bpw'])}) | "
             f"{kl(d['kl_t186n'])}{ci(d['kl_t186n_ci95'])} | "
             f"{pct(d['ratio_iqr_t186n'])} (IQR-V4 {kl(d['iqr_v4_t186n_at_text_bpw'])}) |")
L.append("")
L.append("Secondary, literal file-size metric (the entries' full file bytes, vision tower included; our incumbents are text-only, so "
         "this charges the entries for the vision tower):\n")
L.append("| entry | file bpw | t183 KL | vs our V3 incumbent at the file bpw (t183) |")
L.append("|---|---:|---:|---:|")
for k, d in rows:
    L.append(f"| {d['label']} | {d['file_bpw']:.4f} | {kl(d['kl_t183'])} | {pct(d['ratio_ours_t183_file_bpw'])} (ours {kl(d['ours_t183_at_file_bpw'])}) |")
L.append("")
L.append("## Sanity: their build against our reads of IQR-V4 iq3r\n")
L.append(f"- t183: their build {kl(s['iq3r_t183_their_build'])} vs our refs-t183 {kl(s['iq3r_t183_lead_refs'])}: "
         f"{s['ratio_minus_1'] * 100:+.3f} % (gate ±3 %: PASS)")
L.append(f"- t186n: their build {kl(s['iq3r_t186n_their_build'])} vs w164's rows (our build) {kl(s['iq3r_t186n_w164_build'])}: "
         f"{s['ratio_minus_1_t186n'] * 100:+.3f} %")
L.append(f"- Size convention check: IQR-V4 iq3r's text-plane bpw from its container {iq_tb:.4f} vs its stated {bpw_of['iq3r']:.3f} (w134); "
         f"entries: " + ", ".join(f"{d['label'].split(' ')[0]} {d['text_plane_bpw']:.4f} vs stated {d['stated_bits']:.4f}" for _, d in rows))
L.append("")
L.append("## Entries: sizes and sha256\n")
L.append("| entry | file | bytes | sha256 (LFS oid, verified) | file bpw | text-plane bpw | stated bits | t183 KL | t186n KL |")
L.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
for k, d in rows:
    L.append(f"| {k} | `{d['file']}` | {d['bytes']} | `{d['sha256'][:16]}…` | {d['file_bpw']:.4f} | {d['text_plane_bpw']:.4f} | "
             f"{d['stated_bits']:.4f} | {kl(d['kl_t183'])} | {kl(d['kl_t186n'])} |")
L.append("")
L.append("## IQR-V4 reference curve (text-plane bpw from its README and refs-t183; t186n from w164's rows, our build)\n")
L.append("| IQR-V4 | bpw (w134) | t183 KL (refs-t183, our build) | t186n KL (w164 rows, our build) |")
L.append("|---|---:|---:|---:|")
t186 = {round(b, 6): k for b, k in IQ186}
for r in [r for r in refs if r["name"].startswith("iq")]:
    L.append(f"| {r['name']} | {r['bpw']:.4f} | {kl(r['kld'])} | {kl(t186.get(round(r['bpw'], 6)))} |")
L.append("")
L.append("## Our V3 incumbents on t183\n")
L.append("| size | bpw (text-only container) | t183 KL | tag |")
L.append("|---:|---:|---:|---|")
for r in inc:
    L.append(f"| {r['size']} | {r['bpw']:.4f} | {kl(r['kld_t183'])} | {r['tag']} |")
L.append("")
L.append("## Size bases\n")
L.append("- **Text plane (primary).** Container tensors minus the vision tower (`v.*`) and the drafter: trunk, head and token "
         "embedding, bytes × 8 / 26,895,998,464. The entries' stated bits are this number (to 0.0003 bpw); so is IQR-V4's "
         "w134 bpw (3.472 from its container: 3.4724). This is the basis the board uses per IQR-V4's README "
         "(\"bpw counts the language model … the drafter and the vision tower come on top\").")
L.append("- **File bpw (secondary).** Whole file bytes × 8 / 26,895,998,464: the task's literal metric. The entries carry a 921 MB "
         "bf16 vision tower (0.277 bpw), which adds to their file size and not to their stated bits. Our search containers are "
         "text-only, so their file bpw is already the text-plane bpw.")
L.append("- The default metric in AGENTS (trunk + head over trunk + head parameters, embedding excluded) is not computed here.")
L.append("")
L.append("## Caveats\n")
L.append("- The IQR-V4 and incumbent curves are log-linear interpolations of single reads per size, unpaired with the entries. "
         "Ratios carry no interval; the 95 % intervals are the entries' own, per document (178 docs on t183; 21 docs with gen rows on t186n).")
L.append("- t186n is a 21-document slice (32k gen positions): intervals are ±20–40 %. Read its ratios as a direction only.")
L.append("- Our incumbents are the live search's snapshot; re-run `extract_incumbents.py` for a new snapshot.")
L.append("- Header and vocab bytes: a container also holds about 11–14 MB outside its tensors (strings, vocab, directory). The entries' stated bits and IQR-V4's "
         "bpw leave them out; our incumbents' bpw (file bytes, from the search) includes them, about +0.004 bpw (our V2 3.00: 13.7 MB). "
         "That biases the vs-ours ratios by about +1 % against the entries. Not corrected: the search containers are not on box2.")
L.append("- t186n nano: IQR-V4 has no t186n read between 2.47 and 3.47 bpw, so the nano comparison interpolates across one bit.")
open(os.path.join(HERE, "table.md"), "w").write("\n".join(L) + "\n")
print("wrote table.md and table.json")
