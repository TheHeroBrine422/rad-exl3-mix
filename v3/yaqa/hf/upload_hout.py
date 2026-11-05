#!/usr/bin/env python3
"""upload_hout.py stage|upload [--yes] -- publish the Qwen3.8-27B diagonal output-Hessian collection (#33-24) to one HF
repo (default Hero422/qwen3.8-27b-hout-diag).

stage:  build STAGE (default /data/hfstage/hout-diag) from the collector's output, as reflinks / hard links (no copies):
          snapshots/n{0128..1920}/{sketchb,dy,xw}/<checkpoint key>.safetensors  (+ halves.json under sketchb)
            sketchb = diag of Sketch B's output factor (hdiag-snapN), dy = Σ dy² (hdiag-snapN-dy),
            xw = Σ ||x||² dy² (hdiag-snapN-xw); "hout_diag" = per-sequence mean, ourq.py --hout layout
          block-normalized/<key>.safetensors + mkbn.json  (mkbn.py over the 1,920 snapshot: the V3 estimator)
          README.md (from README_SRC, default v3/yaqa/hf/README.md next to this script; must contain no "TODO")
        and lists file counts and sizes. Nothing leaves the machine.
upload: stage first, then (only with --yes) upload STAGE with upload_large_folder (~18k files: several commits,
        resumable; rerun after a failure). Token from HF_TOKEN (env only; never written to disk by this script).
Env: HOUT_SRC (default /data/yaqa/hdiag), HOUT_BN (default /data/swap27/hout-bn), STAGE, HF_REPO, README_SRC."""
import glob, os, re, shutil, subprocess, sys, time

SRC = os.environ.get("HOUT_SRC", "/data/yaqa/hdiag"); BN = os.environ.get("HOUT_BN", "/data/swap27/hout-bn")
STAGE = os.environ.get("STAGE", "/data/hfstage/hout-diag"); REPO = os.environ.get("HF_REPO", "Hero422/qwen3.8-27b-hout-diag")
README = os.environ.get("README_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"))
KINDS = {"sketchb": "", "dy": "-dy", "xw": "-xw"}

def link(a, b):
    os.makedirs(os.path.dirname(b), exist_ok=True)
    if os.path.exists(b): os.remove(b)
    if subprocess.run(["cp", "--reflink=always", a, b], capture_output=True).returncode: os.link(a, b)

def stage():
    if os.path.exists(STAGE): shutil.rmtree(STAGE)
    snaps = sorted(int(m.group(1)) for d in glob.glob(f"{SRC}/hdiag-snap*") if (m := re.fullmatch(r".*/hdiag-snap(\d+)", d)))
    assert snaps, f"no snapshots in {SRC}"
    for n in snaps:
        for k, sfx in KINDS.items():
            fs = sorted(glob.glob(f"{SRC}/hdiag-snap{n}{sfx}/*.safetensors"))
            assert len(fs) == 400, (n, k, len(fs))
            for f in fs: link(f, f"{STAGE}/snapshots/n{n:04d}/{k}/{os.path.basename(f)}")
        h = f"{SRC}/hdiag-snap{n}/halves.json"
        if os.path.exists(h): link(h, f"{STAGE}/snapshots/n{n:04d}/sketchb/halves.json")
    fs = sorted(glob.glob(f"{BN}/*.safetensors")); assert len(fs) == 400, ("block-normalized", len(fs))
    for f in fs + [f"{BN}/mkbn.json"]: link(f, f"{STAGE}/block-normalized/{os.path.basename(f)}")
    r = open(README).read(); assert "TODO" not in r, f"{README} still has TODO markers"
    open(f"{STAGE}/README.md", "w").write(r)
    n_files = sum(len(f) for _, _, f in os.walk(STAGE))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, f in os.walk(STAGE) for x in f)
    print(f"STAGED {STAGE}: {len(snaps)} snapshots ({snaps[0]}..{snaps[-1]}), {n_files} files, {size / 2**30:.2f} GiB")

def upload():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    t0 = time.time()   # ~18k files: upload_large_folder (several commits, resumable; rerun to continue after a failure)
    api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=STAGE, num_workers=4)
    print(f"{time.strftime('%H:%M')} UPLOADDONE in {(time.time() - t0) / 60:.1f} min -> https://huggingface.co/{REPO}"); return

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("stage", "upload"): sys.exit(__doc__)
    stage()
    if cmd == "upload":
        if "--yes" not in sys.argv: print("dry run: staged only; rerun with --yes to upload"); sys.exit(0)
        upload()
