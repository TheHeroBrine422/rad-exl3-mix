#!/usr/bin/env python3
"""profreport.py JOBDIR... -- the stage profile of rxq quant jobs (log.jsonl "rxq.stages"): one row per (family, K),
seconds per stage (exclusive), the tensor's wall time and peak VRAM. Stage groups for the table:
  load     hessian_load + weight_load + hout_load (+ group_finalize: the group's su draw replayed by a per-tensor job)
  prep     finalize (H scaling, damping, rotation) + regularize (scales, Hadamards)
  scale    scale_search + trellis[scale_search]      (g_scale: trial trellis quants of sampled tiles)
  ldl      scaled_ldl (+ prepare_hout)
  ldlq     ldlq (LDLQ feedback GEMMs / ldlq_2hess)
  trellis  trellis[ldlq] (the Viterbi search of every tile)
  refit    refit (su / sv refit, incl. the CPU fallback when it ran there)
  rest     other (metrics, back-rotation, scales) + pack + save
--md: markdown table; --json: the grouped numbers."""
import collections, json, os, sys

GROUPS = {"load": ("hessian_load", "weight_load", "hout_load", "group_finalize"), "prep": ("finalize", "regularize"),
          "scale": ("scale_search", "trellis[scale_search]"), "ldl": ("scaled_ldl", "prepare_hout"), "ldlq": ("ldlq",),
          "trellis": ("trellis[ldlq]", "trellis[tile_cd]"), "refit": ("refit",), "rest": ("other", "pack", "save")}
ORDER = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "head"]


def rows(dirs):
    out = []
    for d in dirs:
        p = os.path.join(d, "log.jsonl")
        if not os.path.exists(p): continue
        for l in open(p):
            if not l.strip(): continue
            r = json.loads(l); x = r.get("rxq") or {}
            st = dict(x.get("stages") or {}); st["save"] = st.get("save", 0) + x.get("save", 0)
            g = {k: sum(st.get(s, 0) for s in v) for k, v in GROUPS.items()}
            known = {s for v in GROUPS.values() for s in v}
            g["rest"] += sum(v for s, v in st.items() if s not in known)
            out.append({"key": r["key"], "family": x.get("family"), "K": r["K"], "wall": x.get("wall"), "peak_gib": x.get("peak_gib"),
                        "refit_cpu": None, **g})
    return out


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    R = rows(a)
    if "--json" in sys.argv: print(json.dumps(R, indent=1)); return
    by = collections.OrderedDict()
    for r in sorted(R, key=lambda r: (float(r["K"]), ORDER.index(r["family"]) if r["family"] in ORDER else 99)):
        by.setdefault((r["family"], r["K"]), []).append(r)
    cols = list(GROUPS)
    print("| family | K | n | " + " | ".join(cols) + " | wall s | peak GiB |")
    print("|---|---|---|" + "---|" * len(cols) + "---|---|")
    tot = collections.defaultdict(float)
    for (f, K), rs in by.items():
        n = len(rs)
        print(f"| {f} | {K} | {n} | " + " | ".join(f"{sum(r[c] for r in rs) / n:.2f}" for c in cols)
              + f" | {sum(r['wall'] for r in rs) / n:.2f} | {max(r['peak_gib'] or 0 for r in rs):.2f} |")
        for c in cols + ["wall"]: tot[(K, c)] += sum(r[c] for r in rs)
    for K in sorted({k for k, _ in tot}, key=float):
        w = tot[(K, "wall")]
        print(f"| **all** | {K} | | " + " | ".join(f"{tot[(K, c)]:.1f} ({100 * tot[(K, c)] / w:.0f}%)" for c in cols) + f" | {w:.1f} | |")


if __name__ == "__main__":
    main()
