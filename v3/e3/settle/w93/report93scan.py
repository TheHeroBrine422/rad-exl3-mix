#!/usr/bin/env python3
"""report93scan.py DATA [W86JOBS] -- w93 #123 verdict (../w93.md item 1) from the collected scan files:
DATA/scan/hin/*.json, DATA/scan/hout-colo.json, DATA/scan/selftest/selftest.json, DATA/hf-oids.json, DATA/thor-hin.sha,
DATA/thor-hout.sha, DATA/vm-hout.sha (sha256sum of the VM backup of gpubox /data/swap27/hout-bn); W86JOBS = w86's
dry-run jobs.jsonl (its pins). Prints markdown; writes DATA/verdict.json."""
import glob, json, os, sys

D = sys.argv[1]
W86 = sys.argv[2] if len(sys.argv) > 2 else None


def shas(p):
    return {l.split()[1].lstrip("*"): l.split()[0] for l in open(p) if l.strip()}


st = json.load(open(f"{D}/scan/selftest/selftest.json"))
hin = {os.path.basename(f)[:-5] + ".safetensors": json.load(open(f)) for f in sorted(glob.glob(f"{D}/scan/hin/*.json"))}
ho = json.load(open(f"{D}/scan/hout-colo.json"))
hf = json.load(open(f"{D}/hf-oids.json"))
th_in, th_out, vm_out = shas(f"{D}/thor-hin.sha"), shas(f"{D}/thor-hout.sha"), shas(f"{D}/vm-hout.sha")
pins_in, pins_out = {}, {}
if W86:
    for l in open(W86):
        for k, v in json.loads(l)["params"]["pins"].items():
            b = os.path.basename(k)
            if "/hessians/" in k: pins_in[b] = v
            elif "/hout/" in k: pins_out[b] = v

out = ["## #123 integrity scan: verdict", "", f"selftest: {'PASS' if st['PASS'] else 'FAIL'} "
       f"({', '.join(f'{k}: {v['PASS']}' for k, v in st['cases'].items())})", ""]
fails = []
# H_in checks
n = len(hin); bad = {f: o for f, o in hin.items() if not o["PASS"]}
out.append(f"**H_in: {n} files scanned (exhaustive), {n - len(bad)} pass every check, {len(bad)} fail.**")
for f, o in bad.items():
    out.append(f"- FAIL {f}: {'; '.join(o['fails'])}; notes: {'; '.join(o['notes']) or '-'}"); fails.append((f, o["fails"]))
notes = {f: o["notes"] for f, o in hin.items() if o["notes"] and o["PASS"]}
for f, nt in notes.items(): out.append(f"- note {f}: {'; '.join(nt)}")
csm = max(hin.values(), key=lambda o: o["cs_max"])
out.append(f"- Cauchy-Schwarz: max ratio over all files {csm['cs_max']:.6f} ({csm['file']}, {csm['cs_arg']}); entries > 1 + 1e-5: "
           f"{sum(o['cs_gt_1e-5'] for o in hin.values())}; > 1 + 1e-3: {sum(o['cs_gt_1e-3'] for o in hin.values())}; non-finite: "
           f"{sum(o['nonfinite'] for o in hin.values())}")
ch = [o for o in hin.values() if o["cholesky"]]
out.append(f"- damped Cholesky (1e-3 x mean diag): {sum(o['cholesky']['ok'] for o in ch)} / {len(ch)} succeed "
           f"(files that failed an earlier check are not factored); min L_ii^2 / lambda {min(o['cholesky']['min_diag_L2_over_lambda'] for o in ch):.3g}")
zr = [o for o in hin.values() if o["zeros"]]
out.append(f"- exact zeros: {len(zr)} files ({', '.join(o['file'] + ' ' + str(o['zeros']) for o in zr) or 'none'})")
rows = {o["rows"] for o in hin.values()}
out.append(f"- metadata rows: {sorted(rows)}; k: {sorted({o['k'] for o in hin.values()})}; sizes ok (header check)")
# H_in sha
mm = {"hf": [], "dl": [], "thor": [], "pin": []}
for f, o in hin.items():
    if hf["hf_lfs"].get(f, (None,))[0] != o["sha256"]: mm["hf"].append(f)
    if hf["download_meta"].get(f, {}).get("etag") != o["sha256"]: mm["dl"].append(f)
    if th_in.get(f) != o["sha256"]: mm["thor"].append(f)
    if f in pins_in and pins_in[f] != o["sha256"]: mm["pin"].append(f)
missing = sorted(set(hf["hf_lfs"]) - set(hin))
out.append(f"- sha256: HF LFS oid (dataset {hf['repo_sha'][:8]}, {len(hf['hf_lfs'])} files) mismatches {len(mm['hf'])}; download "
           f"metadata mismatches {len(mm['dl'])}; Thor copy mismatches {len(mm['thor'])} ({len(th_in)} hashed); w86 dry-run pins "
           f"{len(pins_in)} compared, mismatches {len(mm['pin'])}; HF files not on colo: {missing or 'none'}")
for k, v in mm.items():
    if v: fails.append((f"sha256 vs {k}", v))
# H_out
hb = [r for r in ho["rows"] if not r["PASS"]]
out += ["", f"**H_out (bn1920): {ho['n_keys']} keys, {ho['n_files']} files, {ho['n_pass']} pass; extra files {ho['extra_files']}.**"]
for r in hb: out.append(f"- FAIL {r['key']}: {r['fails']}"); fails.append((r["key"], r["fails"]))
good = [r for r in ho["rows"] if r["PASS"]]
out.append(f"- min entry {min(r['min'] for r in good):.3g}, max / min within a tensor up to {max(r['max_over_min'] for r in good):.3g}")
mo = {"thor": [], "vm": [], "pin": []}
for r in good:
    f = r["key"] + ".safetensors"
    if th_out.get(f) != r["sha256"]: mo["thor"].append(f)
    if vm_out.get(f) != r["sha256"]: mo["vm"].append(f)
    if f in pins_out and pins_out[f] != r["sha256"]: mo["pin"].append(f)
out.append(f"- sha256: Thor copy mismatches {len(mo['thor'])} ({len(th_out)} hashed); gpubox source (VM backup, {len(vm_out)} files) "
           f"mismatches {len(mo['vm'])}; w86 dry-run pins {len(pins_out)} compared, mismatches {len(mo['pin'])}")
for k, v in mo.items():
    if v: fails.append((f"hout sha256 vs {k}", v))
ok = st["PASS"] and not fails
out += ["", f"**Verdict (rule): {'PASS' if ok else 'FAIL'}** ({len(fails)} failing items)"]
print("\n".join(out))
json.dump({"PASS": ok, "fails": fails, "hin_files": n, "hout_keys": ho["n_keys"], "sha_mismatch_hin": mm, "sha_mismatch_hout": mo},
          open(f"{D}/verdict.json", "w"), indent=1)
