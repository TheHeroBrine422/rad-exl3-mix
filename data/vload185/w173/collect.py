#!/usr/bin/env python3
"""#185 w173: pull the rollout's results from ds8 into the repo (run on the VM; ds8 is reached through ssh-ds8.sh).

  collect.py REPO_DATA_DIR      (default: this script's folder's parent, i.e. data/vload185)

Writes (all small, no rows, no final/ bytes, no IPs):
  rollout.tsv                  one row per level in order.txt: level, K, final bytes, files, copy check, md5 digest, plane
                               result, status (READY / REJECTED / not run), finished (UTC), plane seconds, pack file
  planes/LEVEL.txt             the plane check of a level: header and rad-info / cmp_dumps output
  rollout/LEVEL.md5            md5 of every file of the level's final/ (ds8 copy; equal to box2's, checked at copy time)
  w173/rollout.log             the driver's log (box2's address masked)
  w173/dry/                    the dry identity run's summary and log files (rows stay on ds8, sha256 in the summary)
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SSH = "/workspace/rad-local/bin/ssh-ds8.sh"   # the VM's ds8 helper
BOX2_MASK = re.compile(r"10\.\d+\.\d+\.\d+")


def pull(dst):
    cmd = "cd /data/q27/w173 && tar cf - order.txt rollout.out results planes dry 2>/dev/null"
    with open(os.path.join(dst, "pulled.tar"), "wb") as f:
        subprocess.run([SSH, cmd], stdout=f, check=True, timeout=900)
    with tarfile.open(os.path.join(dst, "pulled.tar")) as t:
        t.extractall(dst)
    os.remove(os.path.join(dst, "pulled.tar"))


def main(argv):
    data = argv[1] if len(argv) > 1 else os.path.abspath(os.path.join(HERE, ".."))
    tmp = tempfile.mkdtemp(prefix="w173-pull-")
    try:
        pull(tmp)
        order = [x.strip() for x in open(os.path.join(tmp, "order.txt")) if x.strip()]
        rows = []
        for lv in order:
            tsv = os.path.join(tmp, "results", f"{lv}.tsv")
            if not os.path.exists(tsv):
                rows.append([lv, "", "", "", "", "", "", "not run", "", "", ""])
                continue
            f = open(tsv).read().rstrip("\n").split("\t")
            rows.append(f[:11] + [""] * (11 - len(f)))
            os.makedirs(os.path.join(data, "rollout"), exist_ok=True)
            shutil.copy(os.path.join(tmp, "results", f"{lv}.md5.ds8"), os.path.join(data, "rollout", f"{lv}.md5"))
            pdir = os.path.join(data, "planes")
            os.makedirs(pdir, exist_ok=True)
            cmp_log = os.path.join(tmp, "planes", f"{lv}.cmp.log")
            body = open(cmp_log).read() if os.path.exists(cmp_log) else "(no cmp log)\n"
            head = (f"# #185 w173 plane check, level {lv} (ds8, no GPU)\n"
                    f"# pack: {f[10]} (live mixes' name); final/: the box2 ladder-src-mad copy, md5 checked at copy time\n"
                    f"# manifest: final_mix.py (336 trunk names, `<weight> final DIR codebook=mad`); reference: rad-info\n"
                    f"# --weights-dump of the pack; compared by cmp_dumps.py; build: w162 e29c765b + 64735054 (rad-info)\n"
                    f"# result: {f[6]}; status {f[7]}; finished {f[8]} UTC\n")
            open(os.path.join(pdir, f"{lv}.txt"), "w").write(head + body)
        hdr = ["level", "K", "final_bytes", "files", "copy_check", "md5_digest", "planes", "status", "finished_utc",
               "plane_s", "pack"]
        with open(os.path.join(data, "rollout.tsv"), "w") as o:
            o.write("\t".join(hdr) + "\n")
            for r in rows:
                o.write("\t".join(r) + "\n")
        log = open(os.path.join(tmp, "rollout.out")).read()
        log = "\n".join(l for l in log.splitlines() if "Permanently added" not in l) + "\n"
        os.makedirs(os.path.join(HERE), exist_ok=True)
        open(os.path.join(HERE, "rollout.log"), "w").write(BOX2_MASK.sub("<box2>", log))
        dsum = os.path.join(tmp, "dry")
        if os.path.isdir(dsum):
            outd = os.path.join(HERE, "dry")
            os.makedirs(outd, exist_ok=True)
            for root, _, files in os.walk(dsum):
                for name in files:
                    if name.endswith((".log", ".txt", ".sha256")) or name == "summary.txt" or name.endswith(".json"):
                        src = os.path.join(root, name)
                        rel = os.path.relpath(src, dsum)
                        dst = os.path.join(outd, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy(src, dst)
        done = sum(1 for r in rows if r[7] == "READY")
        print(f"collected: {len(rows)} levels in order, {done} READY, {sum(1 for r in rows if r[7] == 'REJECTED')} REJECTED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
