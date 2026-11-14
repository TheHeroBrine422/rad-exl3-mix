# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""S8: levels -> packed containers -> mixes (rad-convert, radmix, census: radiance tools, called, not ported).

The text the swap tests and the knapsack hand-off feed those tools, generated here byte for byte as rad-exl3-mix's
shell tools write it (tests/check_mix.py: identical to the swap27 arms' recipe / mix files):
  recipe(level, K, codebook)   rad-convert's recipe for a whole level: one line per trunk family pattern,
                               "blk.*.<fam>.weight exl3i src=LEVEL codebook=<mul1i8|mul1i4|mul1> rate=K" (swap27k.sh
                               packmixk: TRUNK | sed | sort -u | sed)
  level_mix(pack)              radmix's mix file that swaps a whole packed level into a base: "<weight> <pack>" per
                               trunk weight in TRUNK order (swap27.sh TRUNK)
  mix_text(picks, packs)       a knapsack mix: "<weight> <donor.rad>" per picked weight + "act <weight> <mode>" for
                               int-compute modes (radmix's format; rxq.knap writes it from a knap5 solve)
  pack_script(...)             the gpubox commands of one pack -> mix -> census, with the lead's lock rule (conv.lock +
                               ram.lock exclusive for the convert, 11-07)

python -m rxq.mix recipe LEVEL [-K K] [--codebook CB]     (K / codebook from the level's stamp.json by default)
python -m rxq.mix levelmix PACK [--layers 64]"""
import argparse, json, os, sys

from . import qwen35

CB = {"i8": "mul1i8", "i4": "mul1i4", "mul1": "mul1",          # level codebook -> the exl3i importer's codebook=
      "mad": "mad"}      # #177-03: the exl3i importer's mad codebook (radiance qc-exl3-v3-mad, w150's #177-02)


def trunk_names(n_layers=64):
    """swap27.sh TRUNK: the container's trunk weight names, in its order"""
    out = []
    for l in range(n_layers):
        if qwen35.is_full_attention(l): out += [f"blk.{l}.{t}.weight" for t in ("attn_qg", "attn_k", "attn_v", "attn_output")]
        else: out += [f"blk.{l}.{t}.weight" for t in ("ssm_qkv", "ssm_z", "ssm_out")]
        out += [f"blk.{l}.ffn_gate_up.weight", f"blk.{l}.ffn_down.weight"]
    return out


def fmt_k(K):
    """K as the shell tools print it ($K: 4, 2.5)"""
    K = float(K)
    return str(int(K)) if K.is_integer() else repr(K)


def level_info(level):
    """(K, codebook) of a level: ourq's stamp.json (last args) or an rxq-assembled one"""
    e = json.load(open(os.path.join(level, "stamp.json")))[-1]
    if "codebook" in e: return float(e["K"]), e["codebook"]
    return float(e["args"]["K"]), e["args"].get("codebook") or "mul1"


def recipe(level, K, codebook, n_layers=64):
    pats = sorted({"blk.*." + n.split(".", 2)[2] for n in trunk_names(n_layers)})
    return "".join(f"{p} exl3i src={level} codebook={CB[codebook]} rate={fmt_k(K)}\n" for p in pats)


def level_mix(pack, n_layers=64):
    return "".join(f"{n} {pack}\n" for n in trunk_names(n_layers))


def mix_text(picks, packs, base_names=None):
    """picks {weight: (level name, mode)}, packs {level name: donor .rad} -> radmix's mix file text (weights in TRUNK
    order first, then any others sorted); refuses a pick without a donor and modes radmix cannot set (a4)"""
    order = {n: i for i, n in enumerate(trunk_names())}
    lines, acts = [], []
    for w in sorted(picks, key=lambda n: (order.get(n, 1 << 30), n)):
        lv, mode = picks[w]
        if lv not in packs: raise SystemExit(f"{w}: picked {lv}, no packed donor for it")
        lines.append(f"{w} {packs[lv]}\n")
        if mode != "f16":
            if not (mode == "a8t" or (mode.startswith("a8g") and mode[3:].isdigit())):
                raise SystemExit(f"{w}: mode {mode}: radmix sets f16 / a8t / a8g<N> only")
            acts.append(f"act {w} {mode}\n")
    return "".join(lines + acts)


def pack_script(arm, level, K, codebook, base, work="/data/w76mix", install="/data/install-116", home="/data/swap27/home",
                tools="/data/swap27/tools", py="/data/venv-exl3/bin/python", model="models/qwen38-27b-bf16"):
    """the bash lines of one pack (rad-convert) -> radmix into BASE -> census (gpubox paths; swap27k.sh packmixk with
    conv.lock + ram.lock exclusive)"""
    P, M = f"{work}/pack-{arm}.rad", f"{work}/mix-{arm}.rad"
    n = len(trunk_names())
    return f"""set -e; mkdir -p {work}; cd {work}
cat > {arm}.recipe <<'EOF'
{recipe(level, K, codebook).rstrip()}
EOF
( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; cd /workspace; rm -f {P}.part
  RADIANCE_HOME={home} RADIANCE_HOST_DEVICES=1 nice {tools}/memguard.sh 2500 {install}/bin/rad-convert {model} \\
    --recipe {work}/{arm}.recipe --set quantization_config.quant_method=exl3 -o {P}.part > {work}/conv-{arm}.log 2>&1 && mv {P}.part {P} )
cat > {arm}.mix <<'EOF'
{level_mix(P).rstrip()}
EOF
{tools}/radmix {M}.part {base} {work}/{arm}.mix > {work}/mix-{arm}.log 2>&1 && mv {M}.part {M}
{py} -I {tools}/census.py {M} --mix {work}/{arm}.mix --base {base} > {work}/census-{arm}.txt 2>&1
grep -qE "exl3i  *{fmt_k(K)}(\\.0+)?  *{CB[codebook]} *f16  *{n} " {work}/census-{arm}.txt && rm -f {P}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxq.mix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recipe"); r.add_argument("level"); r.add_argument("-K", type=float); r.add_argument("--codebook")
    m = sub.add_parser("levelmix"); m.add_argument("pack"); m.add_argument("--layers", type=int, default=64)
    a = ap.parse_args(argv)
    if a.cmd == "recipe":
        K, cb = (a.K, a.codebook) if a.K and a.codebook else level_info(a.level)
        sys.stdout.write(recipe(a.level, a.K or K, a.codebook or cb))
    else:
        sys.stdout.write(level_mix(a.pack, a.layers))


if __name__ == "__main__":
    main()
