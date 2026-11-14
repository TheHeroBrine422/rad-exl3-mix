#!/usr/bin/env python3
"""mkheld.py OUTDIR -- #176-02 (w159): the held-out mixes for the residual check (RULES.md), written before any of them is
measured: per mix a radmix file m-NAME.mix (base = a uniform mad pack; per-tensor lines from other uniform packs; head from
w158's slim head donor (/data/q27/w158/heads/hd-PLAN.rad); trellis embedding donor; a/b add / drop lines; act a8g128 lines unless the mix says no) and
m-NAME.assign.json ({w: option} for fit17602.py predict, "ab": bf16 / T4 / T5). Seeded, so a rerun writes the same files.
(w159)"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "search", "v3"))
import menu as MN  # noqa: E402

Q = "/data/q27"; W = f"{Q}/w159"
LV = "q27v3-635bac02-k{}-mad"
KS = ["1", "1.125", "1.25", "1.375", "1.5", "1.625", "1.75", "1.875", "2", "2.125", "2.25", "2.375", "2.5", "2.625", "2.75",
      "2.875", "3", "3.125", "3.25", "3.375", "3.5", "3.625", "3.75", "3.875", "4", "4.125", "4.25", "4.375", "4.5", "4.625",
      "4.75", "4.875", "5", "5.25", "5.5", "5.75", "6", "6.25", "6.5", "6.75", "7", "7.5"]


def pack(k):
    k = str(k)
    return f"{Q}/packs-mad/{LV.format(k)}" + ("-s6c57a3d1.rad" if float(k) <= 4.5 else ".rad")


def emb_src(e):
    k = e[1:].rstrip("m")
    return f"{Q}/emb/trellis/emb-t-{'m1' if e.endswith('m') else 'i8'}-K{k}.rad"


OWN_HD = {"u6.5m", "u5.5"}
MIXES = {   # name: (base K, trunk rule, head plan, a/b, embedding, act lines)
    "h1": ("2.5", ("rand", 2.0, 3.5), "u2.75", "T4", "t2.5", True),
    "h2": ("4", ("rand", 3.5, 5.0), "u4.5", "T4", "t4", True),
    "h3": ("6", ("rand", 5.0, 7.5), "u6.5m", "T5", "t6.5m", True),
    "h4": ("3", ("type", {"ffn_gate_up": "2.25", "ffn_down": "2.25", "attn_qg": "4", "attn_k": "4", "attn_v": "4",
                          "attn_output": "4", "ssm_qkv": "3", "ssm_z": "3", "ssm_out": "3"}), "u3", "T4", "base", False),
    "h5": ("4", ("rand", 1.0, 7.5), "u5.5", "T5", "t8m", True),
}


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(17602)
    for name, (bk, rule, head, ab, emb, act) in MIXES.items():
        asg, lines = {}, [f"#base {pack(bk)}"]
        for w in MN.trunk_names():
            t = w.split(".")[2]
            if rule[0] == "rand": k = rng.choice([k for k in KS if rule[1] <= float(k) <= rule[2]])
            else: k = rule[1][t]
            asg[w] = LV.format(k)
            if k != bk: lines.append(f"{w} {pack(k)}")
        asg[MN.HEAD] = head
        # w158's slim census-checked donor; u6.5m / u5.5 were not built yet when ds8 rebooted (10:08 AM CST, w158's donor
        # builder gone): our own head pack (mk17602.sh hd, HEAD_CONVERT=1: rad-convert of the K1.125 pack + the plan's line)
        lines.append(f"{MN.HEAD} {W}/packs/hd-{head}.rad" if head in OWN_HD else f"{MN.HEAD} {Q}/w158/heads/hd-{head}.rad")
        asg[MN.EMB] = emb
        if emb != "base": lines.append(f"{MN.EMB} {emb_src(emb)}")
        asg["ab"] = ab
        for l in range(64):
            if MN.is_full_attention(l): continue
            lines += [f"drop blk.{l}.ssm_a.weight", f"drop blk.{l}.ssm_b.weight",
                      f"add blk.{l}.ssm_ab.weight {Q}/mad/ab/donor-ab-K{ab[1:]}.rad"]
        if act:   # every trunk linear + the added ssm_ab (qwen35_exl3: one activation mode a shared-input group, qkv / z / ab)
            lines += [f"act {w} a8g128" for w in MN.trunk_names()]
            lines += [f"act blk.{l}.ssm_ab.weight a8g128" for l in range(64) if not MN.is_full_attention(l)]
        open(f"{out}/m-{name}.mix", "w").write("\n".join(lines) + "\n")
        json.dump(asg, open(f"{out}/m-{name}.assign.json", "w"), indent=0)
        ks = [float(v.split("-k")[1].split("-")[0]) for w, v in asg.items() if w.startswith("blk.")]
        print(f"m-{name}: base K{bk}, trunk K {min(ks):g}-{max(ks):g} mean {sum(ks) / len(ks):.3f}, head {head}, a/b {ab}, emb {emb}, act {act}, {len(lines)} lines")


if __name__ == "__main__":
    main(sys.argv[1])
