# MACaQ (arXiv 2611.10649, Huawei CSL, 2026-11-07) — status: read 11-11, nothing to adopt for V3

MXFP4 W4A4 PTQ that treats each linear layer alone: online per-channel balance D (α 0.75) + 32-point block Hadamard on
the activations, inverses folded into the weights (no cross-layer transforms); weights RTN to MXFP4 with a scale-exponent
search (+1 / 0 / −1), then **Flip-Refinement**: Hessian-aware greedy coordinate descent moving single FP4 codes to an
adjacent value (up to 512 iterations, ridge 1.0). 2.3% end-to-end transform overhead on Qwen3-32B (vLLM); a Qwen3.8-27B
llama.cpp path (IQ4_XS, had-block 64; 2.74× decode vs FP16). Code: github.com/huawei-csl/MACaQ (Apache-2.0). Same team
as KVarN.

Against our stack:
- **Flip-Refinement = our tile coordinate descent (#77, `rxq/tilecd.py`).** On top of LDLQ + Viterbi trellis codes the
  real gain is small (attn_qg 0.987–0.996 proxy, 27B no-harm 0.9994–0.9997 at ×3.7 quant time), which is why it is on
  attn_qg only. MACaQ's large gains are over RTN, which has no error feedback; nothing new to test.
- **Layer-local transforms:** we already keep every transform inside its layer (suh / svh, input and output Hadamard),
  and quantize layer by layer from Hessians.
- **Activation balancing for A4:** A4 is dropped for V3; our A8's KLD cost is ×1.0014 (#122), so no gain at A8.
- **Competition:** a Qwen3.8-27B IQ4_XS GGUF from it may appear; ~4.25 bpw class, no trellis. Watch the board.
