"""sub1chk92.py A -- ../w92.md "sub-1 job check": the K0.875 / K0.125 i4 jobs of layers 2-3 (A/out/sub1-*): every tensor's
logged sigma (rxq's record: the sigma LDLQ used, sigma_reg_in for the layer-input group) as v3.toml says (K0.875: 0.05 on
the layer-input group, 0.025 on the rest; K0.125: 0.2 everywhere), H_out on (blocks / group), block order on the
readers, tile CD records for q_proj, C9 files for the plain writers, finite proxy errors. Run with pipeline/ on the path."""
import glob, json, math, os, sys
from rxq import qwen35
A = sys.argv[1]; bad = 0
WANT = {"sub1-0.875": (0.05, 0.025), "sub1-0.125": (0.2, 0.2)}
READ = ("ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v", "ffn_gate", "ffn_up")
for tag, (s_in, s_rest) in WANT.items():
    n = 0
    for d in sorted(glob.glob(f"{A}/out/{tag}/L*")):
        tcd = {json.loads(l)["key"] for l in open(f"{d}/tilecd.jsonl")} if os.path.exists(f"{d}/tilecd.jsonl") else set()
        for l in open(f"{d}/log.jsonl"):
            r = json.loads(l); k = r["key"]; fam = r["rxq"]["family"]; n += 1
            hname = qwen35.locate(k)[3]; inp = qwen35.is_input_group(hname)
            want = s_in if inp else s_rest
            checks = {"sigma": r["sigma_reg"] == want, "finite": math.isfinite(r["proxy_err"]), "hout": r["hout"] is True,
                      "blocks": r.get("hout_blocks") == "group", "order": bool(r.get("block_order")) == (fam in READ),
                      "tilecd": (k in tcd) == (fam == "attn_qg"), "c9": ("c9" in r) == os.path.exists(f"{d}/c9-{k}.safetensors") == (fam in ("attn_output", "ssm_out"))}
            ok = all(checks.values()); bad += not ok
            print(f"{tag} {k.split('layers.')[1]:28s} {fam:11s} sigma {r['sigma_reg']} (want {want}) proxy {r['proxy_err']:.4e} "
                  f"c9 {'c9' in r} {'ok' if ok else 'BAD ' + str([c for c, v in checks.items() if not v])} "
                  f"wall {r['rxq']['wall']:.1f}s peak {r['rxq']['peak_gib']} GiB")
    print(f"{tag}: {n} tensors")
print(f"SUB1 {'PASS' if bad == 0 else 'FAIL'} ({bad} bad)")
