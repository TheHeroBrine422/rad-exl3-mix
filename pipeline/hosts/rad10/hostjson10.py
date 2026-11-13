"""hostjson10.py HOST.json [--score DIR] [--back] -- #170: point a box2 / ds8 rxrun.hostd config's eval commands at the
ROCm 10.1 build. score_cmd*: DIR/install/bin/radiance with RADIANCE_HOME=DIR/home (DIR default /data/q27/rad10; the
374503f media build is /data/q27/rad10/r374). pack_cmd: /data/q27/rad10/install/bin/rad-convert with
RADIANCE_HOME=/data/q27/rad10/home-pack (packhome.sh: text-only containers, byte-identical to the 7.14 packs; a3799b5's
qwen35_exl3 cannot pack from the HF checkpoint and 639ad2f's packs carry the bf16 vision tower). Works from the 7.14
config or an already switched one. The first original is kept as HOST.json.bak-714 (never overwritten); --back restores
it. (w130)"""
import json, os, re, shutil, sys

p = sys.argv[1]
if "--back" in sys.argv:
    shutil.copy2(p + ".bak-714", p); print(f"{p}: restored from .bak-714"); sys.exit(0)
d = sys.argv[sys.argv.index("--score") + 1].rstrip("/") if "--score" in sys.argv else "/data/q27/rad10"
j = json.load(open(p))
if not os.path.exists(p + ".bak-714"):
    shutil.copy2(p, p + ".bak-714")
INST = re.compile(r"/data/q27/rad(?:10(?:/r374)?)?/install/bin/")
HOME = re.compile(r"RADIANCE_HOME=/data/q27/rad(?:10(?:/r374)?)?/home(?:-pack)?(?=\s)")
n = 0
for k, v in j.get("eval", {}).items():
    if not (k.endswith("_cmd") or k.endswith("_cmd_progress")) or not isinstance(v, list):
        continue
    pack = k == "pack_cmd"
    inst = "/data/q27/rad10/install/bin/" if pack else d + "/install/bin/"
    home = "RADIANCE_HOME=" + ("/data/q27/rad10/home-pack" if pack else d + "/home")
    for i, a in enumerate(v):
        b = HOME.sub(home, INST.sub(inst, a))
        if b != a:
            v[i] = b; n += 1
json.dump(j, open(p + ".tmp", "w"), indent=1)
os.replace(p + ".tmp", p)
print(f"{p}: {n} command strings changed (score {d}, pack /data/q27/rad10 + home-pack)")
