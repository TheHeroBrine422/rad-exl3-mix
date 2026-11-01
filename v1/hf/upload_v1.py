#!/usr/bin/env python3
"""upload_v1.py DIR [REPO] -- the EXL3-V1 release (turboderp's ladder as radiance containers) into ONE HF repo (default
Hero422/qwen3.8-27b-exl3-turboderp-ladder), in one commit: DIR/*.rad, DIR/recipes/*.recipe, DIR/LICENSE (Qwen3.8's) and
this folder's README.md. Recipes: v1/results/*.recipe with src= pointed at turboderp/Qwen3.8-27B-exl3@<branch>. Retries
the whole commit on failure (chunks already on the Hub are not re-sent). Token from HF_TOKEN (env only).
The 2026-10-31 upload copied each .rad from the first, superseded per-size repos instead (git history: v1/hf before
2026-11-01); the files are the same."""
import os, shutil, sys, time
from huggingface_hub import HfApi

D = sys.argv[1]; REPO = sys.argv[2] if len(sys.argv) > 2 else "Hero422/qwen3.8-27b-exl3-turboderp-ladder"
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), os.path.join(D, "README.md"))
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", exist_ok=True)
for attempt in range(1, 6):
    t0 = time.time()
    try:
        api.upload_folder(repo_id=REPO, repo_type="model", folder_path=D,
                          allow_patterns=["*.rad", "recipes/*.recipe", "LICENSE", "README.md"],
                          commit_message="EXL3-V1: turboderp's EXL3 ladder, 16 containers")
        print(f"{time.strftime('%H:%M')} UPLOADDONE in {(time.time() - t0) / 60:.0f} min", flush=True)
        break
    except Exception as e:
        print(f"{time.strftime('%H:%M')} FAIL attempt {attempt}: {e}", flush=True); time.sleep(120)
