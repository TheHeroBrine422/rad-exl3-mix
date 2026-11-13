# w132 (#172) RULES-value amendment 1, on ds8: M1 picks' eval mixes (Menu.eval_mix on menu133e, made on the VM) ->
# hostd-style .mix files (trunk weights in TRUNK order -> packs/<level>-s<scales[:8]>.rad, then the donors)
import json, os, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxq import mix as M
src, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
for tag, em in json.load(open(src)).items():
    def pkn(n):
        s = json.load(open(f"/data/q27/ladder/{n}/DONE")).get("scales")
        p = f"/data/q27/packs/{n}" + (f"-s{s[:8]}" if s else "") + ".rad"
        assert os.path.exists(p), p; return p
    lines = [f"{w} {pkn(em['levels'][w])}" for w in M.trunk_names()] + [f"{w} {p}" for w, p in em["donors"].items()]
    for w, p in em["donors"].items(): assert os.path.exists(p), p
    open(f"{outdir}/{tag}.mix", "w").write("\n".join(lines) + "\n"); print(tag, len(lines))
