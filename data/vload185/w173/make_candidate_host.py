#!/usr/bin/env python3
"""#185 w173: the candidate ds8 host.json for the final/ path (prepared, NOT applied: the live host.json is the lead's).

  make_candidate_host.py LIVE_HOST_JSON OUT_HOST_JSON            candidate (w173 dirs; for the dry run and the tests)
  make_candidate_host.py --switch LIVE_HOST_JSON OUT_HOST_JSON [--tree DIR]
                                                                 the switch file (SWITCH.md step 3): the live root, mixes,
                                                                 scores and tree are kept (default tree: the live code tree,
                                                                 which must carry hostd-hybrid.patch); the rest as below

Changes from the live file, and nothing else:
  root / mixes / scores   -> /data/q27/w173/... (a candidate's units, manifests and scores never share the live dirs)
  tree                    -> /data/q27/w173/code (the live code tree + hostd-hybrid.patch, applied to a copy)
  eval.final_root         -> /data/q27/finals-mad (the levels' final/ on this host; w162 #185 B)
  eval.final_codebook     -> mad
  eval.virtual_bytes_cmd  -> the candidate build's rad-info --bytes (the served size of a manifest)
  eval.score_cmd(_progress) -> the candidate build's radiance (rad10 fresh build of 64735054, /data/q27/w173/rad10)
  eval.census_virtual_cmd -> census_hybrid.py (w173) with the candidate code's store/census.py, --fast
Left as they are: packs, mix_cmd, pack_cmd, census_cmd (a hybrid candidate is a manifest: hostd uses census_virtual_cmd only),
cards / locks, goldcat, aliases, slots, limits, run_conf.
eval.census_virtual_cmd (w173): census_hybrid.py on the manifest: the final lines against each level's READY marker and md5 list,
the pack lines by census_manifest.py (census.py --fast). hostd refuses a final_root unit without it (fail closed).
The live host has no census at all (census_cmd unset on ds8, 11-15): this candidate adds one, and the lead decides whether the
switch keeps it (SWITCH.md, gate 2).
"""
import json
import sys

W = "/data/q27/w173"
RAD = f"{W}/rad10/install/bin/radiance"
RAD_INFO = f"{W}/rad10/install/bin/rad-info"
HOME = f"{W}/rad10/home"


def main(argv):
    switch, tree, args = False, None, []
    i = 1
    while i < len(argv):
        if argv[i] == "--switch":
            switch = True; i += 1
        elif argv[i] == "--tree":
            tree = argv[i + 1]; i += 2
        else:
            args.append(argv[i]); i += 1
    if len(args) != 2:
        print(__doc__)
        return 2
    live_path, out_path = args
    h = json.load(open(live_path))
    e = h["eval"]
    if not switch:
        h["root"] = f"{W}/multi"
        h["tree"] = f"{W}/code"
        e["mixes"] = f"{W}/mixes"
        e["scores"] = f"{W}/scores"
    else:
        h["tree"] = tree or h["tree"]                     # the live code tree (patched by hostd-hybrid.patch)
    e["final_root"] = "/data/q27/finals-mad"
    e["final_codebook"] = "mad"
    e["virtual_bytes_cmd"] = ["bash", "-c", f"RADIANCE_HOME={HOME} {RAD_INFO} --bytes {{mix}}"]
    e["census_virtual_cmd"] = ["python3", "-I", f"{W}/tools/census_hybrid.py", "{mix}", "--mix", "{mixfile}", "--base", "{base}",
                               "--final-root", e["final_root"], "--codebook", e["final_codebook"], "--census",
                               f"{W}/code/store/census.py", "--fast"]
    old = e["score_cmd"][2]                      # ["bash", "-c", "ROCR_VISIBLE_DEVICES=... radiance ..."]
    assert old.startswith("ROCR_VISIBLE_DEVICES={card} RADIANCE_HOME=/data/q27/rad10-e157b/home /data/q27/rad10-e157b/install/bin/radiance ")
    new = f"ROCR_VISIBLE_DEVICES={{card}} RADIANCE_HOME={HOME} {RAD}" + old[len(
        "ROCR_VISIBLE_DEVICES={card} RADIANCE_HOME=/data/q27/rad10-e157b/home /data/q27/rad10-e157b/install/bin/radiance"):]
    e["score_cmd"][2] = new
    p = e["score_cmd_progress"][2]
    assert p.startswith("ROCR_VISIBLE_DEVICES={card} RADIANCE_HOME=/data/q27/rad10-e157b/home /data/q27/rad10-e157b/install/bin/radiance ")
    e["score_cmd_progress"][2] = f"ROCR_VISIBLE_DEVICES={{card}} RADIANCE_HOME={HOME} {RAD}" + p[len(
        "ROCR_VISIBLE_DEVICES={card} RADIANCE_HOME=/data/q27/rad10-e157b/home /data/q27/rad10-e157b/install/bin/radiance"):]
    with open(out_path, "w") as f:
        json.dump(h, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
