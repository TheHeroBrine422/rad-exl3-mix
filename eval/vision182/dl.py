#!/usr/bin/env python3
# dl.py OUTDIR -- the pinned parquet shards of the #182 vision set (w156)
import sys, os
from huggingface_hub import hf_hub_download, HfApi
out = sys.argv[1]
F = [("HuggingFaceM4/A-OKVQA", "d1b0efa3a4", ["data/validation-00000-of-00001-b2bd0de231b6326a.parquet"]),
     ("lmms-lab/textvqa", "9c0699cd19768ac5ab97568f6b3cbac4c0062884", ["data/validation-00000-of-00003.parquet"]),
     ("lmms-lab/DocVQA", "539088ef8a8ada01ac8e2e6d4e372586748a265e", ["DocVQA/validation-00000-of-00006.parquet", "InfographicVQA/validation-00000-of-00004.parquet"]),
     ("lmms-lab/ChartQA", "9e63b7df15", ["data/test-00000-of-00001.parquet"]),
     ("HuggingFaceM4/WebSight", "b11f8172f8", ["data/train-00000-of-00071-eb722b04b83e13b7.parquet"]),
     ("rootsautomation/ScreenSpot", "0be08781e2", ["data/test-00000-of-00003.parquet"]),
     ("lmms-lab/NLVR2", "a101e1114e", ["data/balanced_dev-00000-of-00001-54ed883ce96f619e.parquet"]),
     ("MMMU/MMMU", "876ce5cb130f7f7e290ce4d9984357737d4db5cf", [f"{s}/validation-00000-of-00001.parquet" for s in
        ["Accounting", "Architecture_and_Engineering", "Biology", "Chemistry", "Computer_Science", "Economics", "Electronics",
         "Energy_and_Power", "Finance", "Geography", "History", "Manage", "Math", "Physics", "Sociology", "Psychology", "Public_Health", "Materials", "Mechanical_Engineering", "Music"]])]
api = HfApi()
for repo, sha, files in F:
    full = api.dataset_info(repo, revision=sha).sha
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision=full, local_dir=os.path.join(out, repo.replace("/", "__")))
        print(repo, full, f, os.path.getsize(p), flush=True)
