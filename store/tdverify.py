#!/usr/bin/env python3
"""tdverify.py REFDIR PACK.rad -- does PACK carry every EXL3 tensor of turboderp's download REFDIR intact?

Codes are compared byte-exact: his tile streams (int16 [kt][nt][16K], 8K little-endian u32 words per tile, symbol p =
stream bits [pK, (p+1)K) MSB-first, at tile position (r, c) = exl3_pos_to_tile(p)) are rebuilt into the pack's canonical
plane (uK [n][k], row-major, LSB-first), the importer's mapping (exl3import/common/exl3_layout.c, exl3_decode.c).
suh / svh are compared byte-exact (copied verbatim). gate_proj + up_proj map to the stacked ffn_gate_up (planes
suh = gate's, suh.1 = up's, svh = gate's then up's). Non-trellis tensors (norms, bf16 a/b, the embedding) are
not checked: they are the checkpoint's own bf16 values, kept separately. Exit 1 on any mismatch or missing tensor."""
import glob, json, os, struct, sys
import numpy as np
from safetensors import safe_open

REF, PACK = sys.argv[1:3]
HDR = struct.Struct("<IIQ5Q" + "Q" * 18); ENT = struct.Struct("<QQQII6qiiiIQQQ2Q"); ENC = 568

def cstr(b, o): return "" if o == 0 else b[o:b.index(b"\0", o)].decode()
f = open(PACK, "rb"); h = HDR.unpack(f.read(HDR.size))
str_off, str_n, _, _, dir_off, dir_n, pl_off, pl_n, enc_off, enc_n = h[8:18]
f.seek(str_off); S = f.read(str_n); f.seek(dir_off); D = f.read(dir_n * ENT.size)
f.seek(pl_off); P = f.read(pl_n * 16); f.seek(enc_off); E = f.read(enc_n * ENC)
pack = {}
for i in range(dir_n):
    e = ENT.unpack_from(D, i * ENT.size); name = cstr(S, e[0]); eb = E[e[3] * ENC:(e[3] + 1) * ENC]
    roles = [eb[56 + 64 * p:56 + 64 * p + 24].split(b"\0")[0].decode() for p in range(e[14])]
    planes = {}
    for p, r in enumerate(roles):
        po, pn = struct.unpack_from("<QQ", P, (e[15] + p) * 16); planes[r] = (po, pn)
    pack[name] = planes
def plane(name, role):
    po, pn = pack[name][role]; f.seek(po); return f.read(pn)

MAP = {"mlp.down_proj": "ffn_down", "self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k",
       "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "linear_attn.in_proj_qkv": "ssm_qkv",
       "linear_attn.in_proj_z": "ssm_z", "linear_attn.out_proj": "ssm_out"}
if os.path.exists(f"{REF}/model.safetensors.index.json"):
    idx = json.load(open(f"{REF}/model.safetensors.index.json"))["weight_map"]
else:                                                  # single-file download
    with safe_open(f"{REF}/model.safetensors", "np") as s_: idx = {k: "model.safetensors" for k in s_.keys()}
src = {}
for k, fn in idx.items():
    if k.endswith(".trellis"): src[k[:-8]] = fn
def get(base, sub):
    with safe_open(f"{REF}/{src[base]}", "np") as s: return s.get_tensor(f"{base}.{sub}")
_off = (0, 1, 8, 9)
PR = np.array([2 * ((p // 8) % 4) + _off[(p % 8) % 4] for p in range(256)])
PC = np.array([(p // 8) // 4 + 8 * ((p % 8) // 4) for p in range(256)])
def canon(t, CHN=256):
    """his trellis int16 [kt, nt, 16K] -> canonical codes plane bytes [16 nt][ceil(16 kt K / 8)], by n-tile chunks"""
    kt, nt, w = t.shape; K = w // 16; out = []
    for a in range(0, nt, CHN):
        u = np.ascontiguousarray(t[:, a:a + CHN]).view(np.uint32)                  # LE pairs -> u32
        bits = np.unpackbits(u.astype(">u4").view(np.uint8), axis=-1, bitorder="big")  # MSB-first stream
        n_ = u.shape[1]
        sym = bits.reshape(kt, n_, 256, K) @ (1 << np.arange(K - 1, -1, -1)).astype(np.uint8)
        T = np.zeros((kt, n_, 16, 16), np.uint8); T[:, :, PR, PC] = sym
        S = T.transpose(1, 3, 0, 2).reshape(n_ * 16, kt * 16)                        # [n][k]
        rb = ((S[..., None] >> np.arange(K, dtype=np.uint8)) & 1).reshape(S.shape[0], -1)
        out.append(np.packbits(rb, axis=1, bitorder="little").tobytes())
    return b"".join(out)
codes_role = None
bad, ok = [], 0
done = set()
mtp = sorted(b for b in src if b.startswith("mtp."))
for base in sorted(src):
    if base in done or base.startswith("mtp."): continue
    if base == "lm_head":
        tgt, parts = "output.weight", [base]
    else:
        if not base.startswith("model.language_model.layers."): bad.append(f"{base}: no mapping"); continue
        m = base.split("layers.")[1]; L, rest = m.split(".", 1)
        if rest in ("mlp.gate_proj", "mlp.up_proj"):
            g, u = base.replace("up_proj", "gate_proj"), base.replace("gate_proj", "up_proj")
            tgt, parts = f"blk.{L}.ffn_gate_up.weight", [g, u]
        elif rest in MAP: tgt, parts = f"blk.{L}.{MAP[rest]}.weight", [base]
        else: bad.append(f"{base}: no mapping"); continue
    done.update(parts)
    if tgt not in pack: bad.append(f"{base}: {tgt} missing in pack"); continue
    pl = pack[tgt]; cr = next(r for r in pl if r.startswith("codes"))
    want = b"".join(canon(get(p, "trellis")) for p in parts)
    have = plane(tgt, cr)
    if have[:len(want)] != want or any(have[len(want):]):
        bad.append(f"{tgt}: codes differ ({len(have)} vs {len(want)} bytes)"); continue
    suh_ok = plane(tgt, "suh") == get(parts[0], "suh").tobytes() and \
        (len(parts) == 1 or plane(tgt, "suh.1") == get(parts[1], "suh").tobytes())
    svh_ok = plane(tgt, "svh") == b"".join(get(p, "svh").tobytes() for p in parts)
    if not (suh_ok and svh_ok): bad.append(f"{tgt}: suh {'ok' if suh_ok else 'DIFF'} svh {'ok' if svh_ok else 'DIFF'}"); continue
    ok += 1
print(f"{os.path.basename(REF)} -> {os.path.basename(PACK)}: {ok} tensors intact, {len(bad)} problems; "
      f"{len(mtp)} MTP trellis tensors NOT in packs (keep separately)")
for b in bad[:20]: print("  ", b)
sys.exit(1 if bad else 0)
