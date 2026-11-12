# moves.py -- mixfiles for the move-noise runs (w123): V2 3.00 base, V2 3.50's version of a tensor group spliced in
import json, re, os
a, b = [json.loads(l) for l in open("logs/radls.jsonl")]
D = "/data/q27/w123/mixes/qwen3.8-27b-exl3-v2-3.50bpw.rad"
d = [k for k in a["w"] if a["w"][k] != b["w"].get(k) and k.startswith("blk.")]
L = lambda k: int(re.match(r"blk\.(\d+)\.", k).group(1))
groups = {f"L{8*i:02d}-{8*i+7:02d}": [k for k in d if 8 * i <= L(k) < 8 * i + 8] for i in range(8)}
groups["ffn_down"] = [k for k in d if "ffn_down" in k]
groups["attn"] = [k for k in d if ".attn_" in k]
groups["L00-03"] = [k for k in d if L(k) < 4]
groups["L60-63"] = [k for k in d if L(k) >= 60]
groups["all"] = d
os.makedirs("mixfiles", exist_ok=True)
for g, ks in groups.items():
    open(f"mixfiles/mv-{g}.mix", "w").write("".join(f"{k} {D}\n" for k in ks))
    print(g, len(ks), sum(b["w"][k] - a["w"][k] for k in ks) // 1000000, "MB")
