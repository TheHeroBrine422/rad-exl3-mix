#!/usr/bin/env python3
"""headx_test.py CONFIG.json GOLDEN POOLS WORK -- CPU self-test of headx.py on a tiny random Qwen3.5 text model (4
layers, width 64, the 27B's vocabulary and layer pattern; CONFIG = the 27B's config.json).
Checks: (1) the run completes; (2) every dumped row's next token (nxt.i32) is the golden's next token for that row
(rows.i64), by the golden's own row bookkeeping (headsens.py); (3) the dumped vectors equal the hidden state of an
independent forward of their document at the mapped position; (4) both Hessians are symmetric PSD, the gen one has
<= the rows of the all-position one, and gen + non-gen = all at equal documents."""
import json, os, subprocess, sys
import numpy as np, torch
from safetensors.torch import save_file
from safetensors import safe_open
cfgf, G, PJ, W = sys.argv[1:5]
os.makedirs(f"{W}/tiny", exist_ok=True)
from transformers import AutoConfig, Qwen3_5ForCausalLM
c = json.load(open(cfgf))["text_config"]
c.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, layer_types=c["layer_types"][:4],
         num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
         linear_key_head_dim=16, linear_value_head_dim=16, model_type="qwen3_5_text", mtp_num_hidden_layers=0)
json.dump(c, open(f"{W}/tiny/config.json", "w"))
torch.manual_seed(0)
tc = AutoConfig.from_pretrained(f"{W}/tiny"); m = Qwen3_5ForCausalLM(tc).eval()
for p_ in m.parameters(): p_.data = p_.data.to(torch.bfloat16)       # parameters only: buffers (rope) stay fp32, as headx loads
sd = {("lm_head.weight" if k == "lm_head.weight" else "model.language_model." + k[len("model."):]): v.contiguous()
      for k, v in m.state_dict().items()}
save_file(sd, f"{W}/tiny/model.safetensors")
json.dump({"weight_map": {k: "model.safetensors" for k in sd}}, open(f"{W}/tiny/model.safetensors.index.json", "w"))
here = os.path.dirname(os.path.abspath(__file__)); O = f"{W}/out"
r = subprocess.run([sys.executable, f"{here}/headx.py", f"{W}/tiny", G, PJ, O, "--tok", "30000", "--dump", "3000",
                    "--per-doc", "200", "--cap", "1024"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-2000:] if r.returncode else "")
ok = r.returncode == 0 and "HEADX DONE" in r.stdout
print("1 run completes:", ok)
meta = json.load(open(f"{G}/kld.json")); toks = np.fromfile(f"{G}/tokens.i32", np.int32)
R = np.fromfile(f"{O}/dump/rows.i64", np.int64); Dc = np.fromfile(f"{O}/dump/doc.i32", np.int32)
NX = np.fromfile(f"{O}/dump/nxt.i32", np.int32)
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in meta["docs"]])])
exp = np.array([toks[off[i] + meta["docs"][i]["score_from"] + 1 + (r_ - meta["docs"][i]["row0"])] for r_, i in zip(R, Dc)])
gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
c2 = bool((exp == NX).all()) and bool(gen[R].all()); ok &= c2
print(f"2 next tokens agree with the golden rows ({len(R)} rows, {len(set(Dc))} docs), all gen: {c2}")
X = torch.from_numpy(np.fromfile(f"{O}/dump/336.x", np.uint16).astype(np.int16)).view(torch.bfloat16).view(-1, 64)
i = int(Dc[0]); d = meta["docs"][i]; L = min(d["tokens"], 1024)
with torch.no_grad():
    h = m.model(input_ids=torch.from_numpy(toks[off[i]:off[i] + L].astype(np.int64))[None]).last_hidden_state[0]
sel = np.nonzero(Dc == i)[0]; pos = R[sel] - d["row0"] + d["score_from"]
xs = X[torch.from_numpy(sel)].float(); e0 = float((h[torch.from_numpy(pos)].float() - xs).norm() / xs.norm())
e1 = float((h[torch.from_numpy(np.maximum(pos - 1, 0))].float() - xs).norm() / xs.norm())
c3 = e0 < 1e-2 and e1 > 0.1; ok &= c3
print(f"3 dumped vectors vs independent forward (doc {i}, {len(sel)} rows): rel err {e0:.1e} at the mapped "
      f"positions, {e1:.2f} one position off: {c3}")
def H(tag):
    with safe_open(f"{O}/hin-{tag}/lm_head.input.safetensors", "pt") as f: up = f.get_tensor("hessian_upper").double(); md = f.metadata()
    k = int(md["k"]); M = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); M[iu[0], iu[1]] = up
    return M + M.triu(1).T, int(md["rows"])
(Hg, ng), (Ha, na) = H("gen"), H("all")
ev = min(float(torch.linalg.eigvalsh(Hg)[0]), float(torch.linalg.eigvalsh(Ha - Hg)[0]))
c4 = ng <= na and ev > -1e-6 * float(Ha.diagonal().max()); ok &= c4
print(f"4 gen rows {ng} <= all {na}; min eig of Hg and of Ha - Hg {ev:.2e}: {c4}")
print(json.load(open(f"{O}/headx.json")))
print("HEADX SELFTEST", "PASS" if ok else "FAIL")
