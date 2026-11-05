#!/usr/bin/env python3
"""upload_hout.py stage|upload [--yes] -- publish the Qwen3.8-27B diagonal output-Hessian collection (#33-24) to one HF
repo (default Hero422/qwen3.8-27b-hout-diag).

stage:  build STAGE (default /data/hfstage/hout-diag) from the collector's output, as reflinks / hard links (no copies):
          snapshots/n{0128..1920}-{sketchb,dy,xw}.safetensors (one tensor per linear, named by checkpoint key; + -sketchb-halves.json)
            sketchb = diag of Sketch B's output factor (hdiag-snapN), dy = Σ dy² (hdiag-snapN-dy),
            xw = Σ ||x||² dy² (hdiag-snapN-xw); "hout_diag" = per-sequence mean, ourq.py --hout layout
          block-normalized.safetensors + block-normalized-mkbn.json  (mkbn.py over the 1,920 snapshot: the V3 estimator)
          README.md (from README_SRC, default v3/yaqa/hf/README.md next to this script; must contain no "TODO")
        and lists file counts and sizes. Nothing leaves the machine.
upload: stage first, then (only with --yes) upload STAGE in one commit (upload_folder; retried). Token from HF_TOKEN (env only; never written to disk by this script).
Env: HOUT_SRC (default /data/yaqa/hdiag), HOUT_BN (default /data/swap27/hout-bn), STAGE, HF_REPO, README_SRC.
Later collections in the same repo (e.g. self-calibration, #91): HOUT_PARTS = "PREFIX=DIR;..." adds collector outputs
(hdiag-snapN[-dy|-xw] dirs) as PREFIX/nNNNN/{sketchb,dy,xw}/, HOUT_DIRS = "NAME=DIR;..." adds plain directories of
.safetensors (+ .json) as NAME/ (e.g. "selfgen/block-normalized=...;blend-w0.25=..."). Files already on the Hub are
not re-sent, so re-uploading the whole stage with new parts costs only the new files."""
import glob, os, re, shutil, subprocess, sys, time

SRC = os.environ.get("HOUT_SRC", "/data/yaqa/hdiag"); BN = os.environ.get("HOUT_BN", "/data/swap27/hout-bn")
STAGE = os.environ.get("STAGE", "/data/hfstage/hout-diag"); REPO = os.environ.get("HF_REPO", "Hero422/qwen3.8-27b-hout-diag")
README = os.environ.get("README_SRC", os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))), "README.md"))
KINDS = {"sketchb": "", "dy": "-dy", "xw": "-xw"}

def link(a, b):
    os.makedirs(os.path.dirname(b), exist_ok=True)
    if os.path.exists(b): os.remove(b)
    if subprocess.run(["cp", "--reflink=always", a, b], capture_output=True).returncode: os.link(a, b)

def merge(files, out, meta):
    """one safetensors with every per-tensor file's "hout_diag" under its checkpoint key"""
    from safetensors import safe_open
    from safetensors.torch import save_file
    d = {}
    for f in files:
        with safe_open(f, "pt") as x: d[os.path.basename(f)[:-len(".safetensors")]] = x.get_tensor("hout_diag").contiguous()
    os.makedirs(os.path.dirname(out), exist_ok=True); save_file(d, out, metadata=meta)

def stage():
    if os.path.exists(STAGE): shutil.rmtree(STAGE)
    parts = [("snapshots", SRC)] + [tuple(x.split("=", 1)) for x in os.environ.get("HOUT_PARTS", "").split(";") if x]
    dirs = [("block-normalized", BN)] + [tuple(x.split("=", 1)) for x in os.environ.get("HOUT_DIRS", "").split(";") if x]
    for pre, src in parts:
        snaps = sorted(int(m.group(1)) for d in glob.glob(f"{src}/hdiag-snap*") if (m := re.fullmatch(r".*/hdiag-snap(\d+)", d)))
        assert snaps, f"no snapshots in {src}"
        for n in snaps:
            for k, sfx in KINDS.items():
                fs = sorted(glob.glob(f"{src}/hdiag-snap{n}{sfx}/*.safetensors"))
                assert len(fs) == 400, (src, n, k, len(fs))
                merge(fs, f"{STAGE}/{pre}/n{n:04d}-{k}.safetensors", {"estimator": k, "sequences": str(n)})
            h = f"{src}/hdiag-snap{n}/halves.json"
            if os.path.exists(h): link(h, f"{STAGE}/{pre}/n{n:04d}-sketchb-halves.json")
        print(f"  {pre}: {len(snaps)} snapshots ({snaps[0]}..{snaps[-1]}) from {src}")
    for name, src in dirs:
        fs = sorted(glob.glob(f"{src}/*.safetensors")); assert len(fs) == 400, (name, len(fs))
        merge(fs, f"{STAGE}/{name}.safetensors", {"estimator": name})
        for f in sorted(glob.glob(f"{src}/*.json")): link(f, f"{STAGE}/{name}-{os.path.basename(f)}")
        print(f"  {name}: {len(fs)} tensors from {src}")
    r = open(README).read(); assert "TODO" not in r, f"{README} still has TODO markers"
    open(f"{STAGE}/README.md", "w").write(r)
    n_files = sum(len(f) for _, _, f in os.walk(STAGE))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, f in os.walk(STAGE) for x in f)
    print(f"STAGED {STAGE}: {n_files} files, {size / 2**30:.2f} GiB")

def upload():
    if not os.path.exists(STAGE): sys.exit(f"{STAGE} not staged")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    for attempt in range(1, 6):   # ~60 files: one commit; chunks already on the Hub are not re-sent on a retry
        t0 = time.time()
        try:
            api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=STAGE,
                              commit_message="Qwen3.8-27B diagonal output Hessians (Sketch B / dy / xw, 128..1920 sequences)")
            print(f"{time.strftime('%H:%M')} UPLOADDONE in {(time.time() - t0) / 60:.1f} min -> https://huggingface.co/datasets/{REPO}"); return
        except Exception as e:
            print(f"{time.strftime('%H:%M')} FAIL attempt {attempt}: {e}", flush=True); time.sleep(60)
    sys.exit(1)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("stage", "upload"): sys.exit(__doc__)
    stage()
    if cmd == "upload":
        if "--yes" not in sys.argv: print("dry run: staged only; rerun with --yes to upload"); sys.exit(0)
        upload()
