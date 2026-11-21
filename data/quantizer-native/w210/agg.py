#!/usr/bin/env python3
"""agg.py -- w210's tables from raw/ (pull.sh copies it from box2:/data/q27/w210/out). Prints markdown."""
import glob, json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
ORDER = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
STG = ["prep", "gscale", "factor", "viterbi", "feedback", "cd", "refit", "pack"]

def rows(path):
    return [json.loads(l) for l in open(path) if l.startswith("{")] if os.path.exists(path) else []

def tool(tag):
    """{(tensor, K): {proxy, record, sec{}, ...}} of an exl3v3_tool tensor run directory (base = w209's build: one line a K)"""
    out = {}
    for t in ORDER:
        last = None
        for d in rows(f"{R}/{tag}/{t}.jsonl"):
            if "batch_of" in d: out[(t, str(d["K"]))]["sec"] = d["sec"]
            else:
                out[(t, str(d["K"]))] = {"proxy": d["proxy_after_refit"], "record": d["proxy_record"], "sec": d.get("sec"), "n": d["n"], "k": d["k"],
                                         "p2s": d.get("proxy_2s"), "cd": d.get("cd_ratio"), "g": d["g_scale"]}
    return out

def ref(tag):
    """{tensor: record} of an rxq run (proxy.jsonl)"""
    return {d["key"].split("layers.3.")[1]: d for d in rows(f"{R}/ref/{tag}.jsonl") if "layers.3." in d["key"]}

def profile():
    base, new = tool("base"), tool("plain")
    print("### layer 3, plain settings (w209's): host pipeline (w209, `base`) vs the card pipeline (`plain`), seconds a stage\n")
    print("| tensor | n x k | K | build | total | prep | scale search | Hessian factor | tile search | LDLQ feedback | refit | pack | load (file) | record write | proxy |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {}
    for t in ORDER:
        for K in ("2", "4"):
            for nm, d in (("host", base), ("card", new)):
                x = d.get((t, K))
                if not x or not x["sec"]: continue
                s = x["sec"]
                print(f"| 3.{t} | {x['n']}x{x['k']} | {K} | {nm} | {s['quantize']:.2f} | {s['prep']:.2f} | {s['gscale']:.2f} | {s['factor']:.2f} | {s['viterbi']:.2f} | "
                      f"{s['feedback']:.2f} | {s['refit']:.2f} | {s['pack']:.2f} | {s['load']:.2f} | {s['write']:.3f} | {x['proxy']:.4e} |")
                a = tot.setdefault((K, nm), {k: 0.0 for k in ["quantize"] + STG + ["load", "write"]})
                for k in a: a[k] += s.get(k, 0.0)
    print("\n| K | build | layer s | prep | scale search | Hessian factor | tile search | LDLQ feedback | refit | pack | not the tile search | load | write |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (K, nm), a in sorted(tot.items()):
        print(f"| {K} | {nm} | {a['quantize']:.1f} | {a['prep']:.2f} | {a['gscale']:.2f} | {a['factor']:.2f} | {a['viterbi']:.2f} | {a['feedback']:.2f} | {a['refit']:.2f} | "
              f"{a['pack']:.2f} | {a['quantize'] - a['viterbi']:.1f} | {a['load']:.2f} | {a['write']:.2f} |")
    return tot

def ident():
    p = f"{R}/ident/ident.txt"
    if not os.path.exists(p): return
    L = [l.split() for l in open(p)]
    print(f"\n### card pipeline vs w209's host pipeline, layer 3, K2 / K4: {sum(l[3] == 'identical' for l in L)} of {len(L)} files byte-identical "
          f"(record, suh, svh of {len(L) // 6} tensors x 2 rates; `raw/ident/ident.txt`)\n")

def syn():
    print("### synthetic expert batches (random weights, random SPD Hessians), one card (card 2), `exl3v3_tool batch`\n")
    print("| run | matrix n x k | K | matrices | batch | Hessians | pipe | wall s | matrices/s | values/s | tiles/s | card s | tile search | scale search | refit | factor | prep | feedback | pack | reader s | writer s | proxy mean |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sh = {0: "one pooled", 1: "one a matrix", 2: "one a pair"}
    out = {}
    for f in sorted(glob.glob(f"{R}/syn/*.jsonl")):
        for d in rows(f):
            c = d["card_sec"]; tag = os.path.basename(f)[:-6]
            print(f"| {tag} | {d['n']}x{d['k']} | {d['K']} | {d['matrices']} | {d['B']} | {sh[d['share']]} | {'yes' if d['pipe'] else 'no'} | {d['wall']:.1f} | "
                  f"{d['matrices_per_s']:.3f} | {d['values_per_s'] / 1e6:.2f}M | {d['tiles_per_s'] / 1e3:.1f}k | {c['quantize']:.1f} | {c['viterbi']:.1f} | {c['gscale']:.1f} | "
                  f"{c['refit']:.1f} | {c['factor']:.1f} | {c['prep']:.1f} | {c['feedback']:.1f} | {c['pack']:.1f} | {d['read_sec']:.1f} | {d['write_sec']:.1f} | {d['proxy_mean']:.4e} |")
            out[(tag, str(d["K"]))] = d
    return out

def extrap(tot, sy):
    print("\n### extrapolations\n")
    for K in ("2", "4"):
        if (K, "card") in tot:
            c, h = tot[(K, "card")]["quantize"], tot.get((K, "host"), {}).get("quantize")
            print(f"- 27B trunk (64 layers ~ layer 3), one level at K{K}, plain settings: card pipeline {c * 64 / 60:.0f} min" + (f" (w209's host pipeline, same session: {h * 64 / 60:.0f} min)" if h else ""))
    tot_tiles = 15360 * 3 * (2304 // 16) * (5120 // 16)
    for K in ("2", "4"):
        gus = [sy[(t, K)]["matrices_per_s"] for t in ("gu-B16-pair", "gu-B16-pair-r") if (t, K) in sy]
        dns = [sy[(t, K)]["matrices_per_s"] for t in ("down-B16", "down-B16-r") if (t, K) in sy]
        if gus and dns:
            gu, dn = sum(gus) / len(gus), sum(dns) / len(dns)       # the mean of the two passes
            per = 2 / gu + 1 / dn                                   # an expert: gate + up (one Hessian) and down
            srch = sum(sy[(t, K)]["card_sec"]["viterbi"] / sy[(t, K)]["matrices"] for t in ("gu-B16-pair-r", "gu-B16-pair-r", "down-B16-r") if (t, K) in sy)
            print(f"- DS4.1-F routed experts at K{K} (15,360 x 3 matrices, {tot_tiles / 1e9:.2f}B tiles): an expert (gate + up sharing a Hessian, down with its own) = "
                  f"{per:.2f} card-s, of which {srch:.2f} s tile search (gate / up {gu:.2f}, down {dn:.2f} matrices/s, the mean of two passes of 128, batches of 16, with the reader / writer threads) "
                  f"-> {15360 * per / 3600:.1f} card-h = **{15360 * per / 3600 / 4:.1f} h on 4 cards**, factorisations, scale searches and refits included "
                  f"(the search alone at this rate: {15360 * srch / 3600 / 4:.1f} h; w209's search-only line at the full-batch rate: 1.7-1.9 h). "
                  f"The ~7B trunk weights add ~{7e9 / 372.2e6 * tot.get((K, 'card'), {}).get('quantize', 0) / 3600 / 4:.2f} h at layer 3's rate. Not in it: reading 290 GB of weights and the "
                  f"per-expert Hessians from disk (the synthetic reader copies from memory), more than one rate")

def lay(tag, K):
    return {t: tool(tag).get((t, K)) for t in ORDER}

def settings():
    print("\n### layer 3 with the release's settings: new (`exl3v3_tool`) vs the release pipeline (rxq), same card\n")
    print("| settings | K | tensor | new proxy | rxq proxy | new / rxq - 1 | new 2-sided proxy | rxq 2-sided | new s | of which search | CD | rxq s | new / rxq time |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summ = []
    for mode in ("plain", "ladder", "release"):
        for K in ("2", "4"):
            new, old = lay(mode, K), ref(f"{mode}w-K{K}")   # the w runs: a warm-up tensor first
            tn = to = 0; lr = []; ts = 0
            for t in ORDER:
                a, b = new.get(t), old.get(t)
                if not a or not b or not a["sec"]: continue
                import math
                s = a["sec"]; st = (b.get("rxq", {}).get("stages") or {})
                bq = b["s"] - sum(v for kk, v in st.items() if kk in ("hessian_load", "weight_load", "hout_load"))   # without the file loads, as the new side
                tn += s["quantize"]; to += bq; ts += s["viterbi"]; lr.append(math.log(a["proxy"] / b["proxy_err"]))
                p2 = f"{a['p2s']:.4e}" if a.get("p2s") else "—"; q2 = f"{b['proxy_err_2s']:.4e}" if "proxy_err_2s" in b else "—"
                print(f"| {mode} | {K} | 3.{t} | {a['proxy']:.4e} | {b['proxy_err']:.4e} | {100 * (a['proxy'] / b['proxy_err'] - 1):+.2f}% | {p2} | {q2} | {s['quantize']:.1f} | "
                      f"{s['viterbi']:.1f} | {s['cd']:.1f} | {bq:.1f} | {s['quantize'] / bq:.2f}x |")
            if lr: summ.append((mode, K, tn, ts, to, 100 * (math.exp(sum(lr) / len(lr)) - 1), len(lr)))
    print("\n| settings | K | tensors | new layer s | of which tile search | rxq layer s | new / rxq time | geo-mean proxy new / rxq |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, K, tn, ts, to, g, n in summ:
        print(f"| {mode} | {K} | {n} | {tn:.1f} | {ts:.1f} | {to:.1f} | {tn / to:.2f}x | {g:+.2f} % |")
    return summ

def extras():
    print("\n### the extras, each against the release pipeline (rxq) on the same tensor (different sign draws)\n")
    E = lambda t: [d for d in rows(f"{R}/extras/{t}.jsonl")]
    one = lambda t: next((d for d in E(t) if "batch_of" not in d), None)
    sec = lambda t: next((d["sec"] for d in E(t) if "batch_of" in d), None)
    print("| extra | tensor | K | new: off -> on | new on / off | rxq: off -> on | rxq on / off | new on / rxq on - 1 | note |")
    print("|---|---|---|---|---|---|---|---|---|")
    def line(name, tensor, K, on, off, ron, roff, key="proxy_after_refit", rkey="proxy_err", note=""):
        if not on: return
        a = on[key]; b = off[key] if off else None; c = ron[rkey] if ron else None; d = roff[rkey] if roff else None
        f = lambda x: f"{x:.4e}" if x is not None else "—"
        print(f"| {name} | 3.{tensor} | {K} | {f(b)} -> {f(a)} | {a / b:.4f} | {f(d)} -> {f(c)} | {(c / d if c and d else float('nan')):.4f} | "
              f"{(100 * (a / c - 1) if c else float('nan')):+.2f}% | {note} |" if b else
              f"| {name} | 3.{tensor} | {K} | {f(a)} | — | {f(c)} | — | {(100 * (a / c - 1) if c else float('nan')):+.2f}% | {note} |")
    lad, rel, cd, ho = ref("ladder-K4"), ref("release-K4"), ref("cd-K4"), ref("hout-K4")
    tcd = {d["key"].split("layers.3.")[1]: d for d in rows(f"{R}/ref/cd-K4.tilecd.jsonl")}
    for s, t in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj")):
        x = one(f"cd-{s}")
        note = f"CD objective after / before: new {x['cd_ratio']:.4f}, rxq {tcd[t]['ratio']:.4f}; CD time new {sec(f'cd-{s}')['cd']:.1f} s" if x and t in tcd else ""
        line("tile CD 3 (no H_out)", t, 4, x, one(f"nocd-{s}"), cd.get(t), lad.get(t), note=note)
    for s, t in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("o", "self_attn.o_proj"), ("g", "mlp.gate_proj")):
        off = one(f"nohout-{s}") or one(f"nocd-{s}")
        line("H_out two-sided LDLQ: one-sided proxy", t, 4, one(f"hout-{s}"), off, ho.get(t), lad.get(t))
        x = one(f"hout-{s}")
        if x and t in ho and "proxy_err_2s" in ho[t]:
            print(f"| H_out two-sided LDLQ: two-sided proxy | 3.{t} | 4 | {x['proxy_2s']:.4e} | — | {ho[t]['proxy_err_2s']:.4e} | — | not comparable | scaled Hessian: new reports tr(E^T H E M) in the metric LDLQ minimised (the scaled one), rxq in the unscaled rotated H; the next rows have one definition |")
    hn = ref("houtns-K4")
    for sfx, t in (("k", "self_attn.k_proj"), ("o", "self_attn.o_proj")):
        x = one(f"houtns-{sfx}")
        if x and t in hn:
            print(f"| H_out, no scaled Hessian: one-sided / two-sided proxy | 3.{t} | 4 | {x['proxy_after_refit']:.4e} / {x['proxy_2s']:.4e} | — | "
                  f"{hn[t]['proxy_err']:.4e} / {hn[t].get('proxy_err_2s', float('nan')):.4e} | — | {100 * (x['proxy_after_refit'] / hn[t]['proxy_err'] - 1):+.2f}% / "
                  f"{100 * (x['proxy_2s'] / hn[t].get('proxy_err_2s', float('nan')) - 1):+.2f}% | the two-sided proxy has one definition here |")
    x = one("houtcd-q"); t = "self_attn.q_proj"
    if x and t in rel:
        print(f"| H_out + tile CD 3 (the release's q) | 3.{t} | 4 | {x['proxy_after_refit']:.4e} | — | {rel[t]['proxy_err']:.4e} | — | {100 * (x['proxy_after_refit'] / rel[t]['proxy_err'] - 1):+.2f}% | "
              f"CD on the two-sided objective: after / before new {x['cd_ratio']:.4f}, rxq {next((d['ratio'] for d in rows(f'{R}/ref/release-K4.tilecd.jsonl') if d['key'].endswith(t)), float('nan')):.4f}; CD time new {sec('houtcd-q')['cd']:.1f} s |")
    rc, rn = ref("clamp-K0.125"), ref("noclamp-K0.125")
    for s, t in (("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj")):
        x = one(f"clamp-{s}")
        note = f"scales replaced: new {x['clamped']} (rxq's log has no count)" if x and t in rc else ""
        line("scale clamp", t, 0.125, x, one(f"noclamp-{s}"), rc.get(t), rn.get(t), note=note)
    print()
    for t in ("r16-v", "nor16-v", "r16-o"):
        x = one(t)
        if x: print(f"- R16 `{t}`: proxy {x['proxy_after_refit']:.4e}, the record decoded by the oracle (stored row order) {x['proxy_record']:.4e}")
    for t in ("stack-gu", "stack-gu-sameseed"):
        for d in E(t):
            if "batch_of" in d: print(f"- stacked gate_up `{t}` K{d['K']}: the stacked record (suh.1) through `exl3i_decode_rows` vs the two halves decoded alone: max |diff| = {d['stack_maxdiff']:g}; "
                                      f"batch of 2 in {d['sec']['quantize']:.1f} s (factor {d['sec']['factor']:.2f} s)")
    for d in rows(f"{R}/extras/swj.jsonl"): print("- swj: `" + json.dumps(d) + "`")

def low_rates():
    print("\n### the scale search's g and the proxy at low rates, new vs rxq (3.k_proj, 3.v_proj; ladder settings, the release's sub-1 dampings)\n")
    print("| tensor | K | new g | rxq g | new / rxq - 1 | new proxy | rxq proxy | new / rxq - 1 | new g, f32 search | new g, another seed |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    one = lambda p: next((d for d in rows(p) if "batch_of" not in d), None)
    for K in ("0.125", "0.25", "0.5", "1", "2", "4"):
        rr = ref({"0.125": "clamp-K0.125", "2": "ladderw-K2", "4": "ladderw-K4"}.get(K, f"sub1-K{K}"))
        for sfx, t in (("k", "self_attn.k_proj"), ("v", "self_attn.v_proj")):
            x = one(f"{R}/extras/clamp-{sfx}.jsonl") if K == "0.125" else one(f"{R}/sub1/new-{sfx}-K{K}.jsonl") if K in ("0.25", "0.5", "1") else None
            if x is None and (t, K) in tool("ladder"):
                y = tool("ladder")[(t, K)]; x = {"g_scale": y["g"], "proxy_after_refit": y["proxy"]}
            b = rr.get(t)
            if not x or not b: continue
            f = one(f"{R}/sub1/new-{sfx}-K0.125-f32.jsonl") if K == "0.125" else None
            s7 = one(f"{R}/sub1/new-{sfx}-K0.125-seed7.jsonl") if K == "0.125" else None
            print(f"| 3.{t} | {K} | {x['g_scale']:.4f} | {b['g_scale']:.4f} | {100 * (x['g_scale'] / b['g_scale'] - 1):+.2f}% | {x['proxy_after_refit']:.4e} | {b['proxy_err']:.4e} | "
                  f"{100 * (x['proxy_after_refit'] / b['proxy_err'] - 1):+.2f}% | {f['g_scale'] if f else float('nan'):.4f} | {s7['g_scale'] if s7 else float('nan'):.4f} |")

def scale_sample():
    print("\n### `scale_sample 4` (the scale search on a quarter of its sample tiles) vs the default, layer 3, ladder settings\n")
    print("| tensor | K | proxy default | proxy sample 4 | sample 4 / default - 1 | g_scale default / sample 4 | scale search s default -> sample 4 | tensor s default -> sample 4 |")
    print("|---|---|---|---|---|---|---|---|")
    lad = tool("ladder")
    for t in ORDER:
        x = {}
        for d in rows(f"{R}/final/ss4.{t}.jsonl"):
            if "batch_of" in d: x[str(d["K"])]["sec"] = d["sec"]
            else: x[str(d["K"])] = {"proxy": d["proxy_after_refit"], "g": d["g_scale"]}
        for K in ("2", "4"):
            a, b = lad.get((t, K)), x.get(K)
            if a and b and a["sec"]:
                print(f"| 3.{t} | {K} | {a['proxy']:.4e} | {b['proxy']:.4e} | {100 * (b['proxy'] / a['proxy'] - 1):+.3f}% | {a['g']:.4f} / {b['g']:.4f} | "
                      f"{a['sec']['gscale']:.2f} -> {b['sec']['gscale']:.2f} | {a['sec']['quantize']:.2f} -> {b['sec']['quantize']:.2f} |")
    p = f"{R}/final/ident2.txt"
    if os.path.exists(p):
        L = [l.split() for l in open(p)]
        print(f"\n- the last commit's build against w209's host pipeline again (3.k_proj, 3.gate_proj, K2 / K4, `--exl3-drift` = w209's drift; the drift tables agree above K1 anyway): "
              f"{sum(l[3] == 'identical' for l in L)} of {len(L)} files byte-identical (`raw/final/ident2.txt`)")

if __name__ == "__main__":
    tot = profile(); ident(); sy = syn(); extrap(tot, sy); settings(); extras(); low_rates(); scale_sample()
