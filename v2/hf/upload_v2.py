#!/usr/bin/env python3
"""upload_v2.py DIR [REPO] -- upload the EXL3-V2 release containers (DIR/*.rad) to one HF repo (default
Hero422/qwen3.8-27b-exl3-v2-rad) in ONE commit (upload_folder; the xet backend uploads the files' chunks in parallel,
more streams with HF_XET_HIGH_PERFORMANCE=1). Retries the whole commit on failure: chunks already on the Hub are not
re-sent, so a retry costs only hashing. Token from HF_TOKEN (env only)."""
import os, sys, time
from huggingface_hub import HfApi

D = sys.argv[1]; REPO = sys.argv[2] if len(sys.argv) > 2 else "Hero422/qwen3.8-27b-exl3-v2-rad"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", exist_ok=True)
for attempt in range(1, 6):
    t0 = time.time()
    try:
        api.upload_folder(repo_id=REPO, repo_type="model", folder_path=D, allow_patterns=["*.rad"],
                          commit_message="EXL3-V2 containers, 1.5-9.0 bpw")
        print(f"{time.strftime('%H:%M')} UPLOADDONE in {(time.time() - t0) / 60:.0f} min", flush=True)
        break
    except Exception as e:
        print(f"{time.strftime('%H:%M')} FAIL attempt {attempt}: {e}", flush=True); time.sleep(120)
