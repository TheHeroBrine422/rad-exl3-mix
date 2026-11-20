# cb208 — codebook ALU bench for routed experts (w208, 2026-11-20)

What each codebook's decode costs inside an int8 (w8a8) / int4 (w4a4) expert GEMV (M = 1, 2, 4, 8 tokens) and a WMMA
prefill GEMM (M = 256) on gfx1201, at integer and fractional K, against the same kernel without a codebook.
Shapes: DS4.1-F routed expert, gate_up 5120 → 4608 and down 2304 → 5120. Results and findings: `RESULTS.md`
(`python3 mkresults.py` rebuilds it), raw lines in `results.jsonl` (dad1, RX 9070 XT) and `results-box2.jsonl` (box2 card 2, R9700).

## Files

| file | what |
|---|---|
| `cb208.hip` | the bench: candidates, GEMV and GEMM kernels, host oracle, timing. One binary a candidate (`-DCAND=`) |
| `gen_tables.py` → `tables.h` | fixed-point maps of the rounded computed codes, CPTC tables (generated, committed) |
| `build.sh` | builds every candidate (`cb208_<Cand>`) |
| `run.sh` | runs every candidate, one at a time, each under the card lock; rerun-safe |
| `report.py` | tables from result lines (median over passes of each pass's median) |
| `asmcount.py` → `asm.jsonl` | instruction counts of each candidate's GEMV inner loop from the device assembly |
| `merge.py` | how dad1's passes were put together into `results.jsonl` (see below) |
| `mkresults.py`, `RESULTS.head.md` → `RESULTS.md` | findings with their numbers filled in from the result files + `report.py`'s tables |
| `results.jsonl` | dad1 (RX 9070 XT): three timing passes, two decode passes, the exactness pass |
| `results-v1-mul_lo.jsonl` | dad1, first build of the CPTC / mul1-map candidates (the `v_mul_lo_u32` forms) |
| `results-box2.jsonl` | box2 card 2 (R9700): two timing passes, two decode passes, tiny-shape exactness |

## Build and run

```sh
# dad1 (RX 9070 XT). The pip ROCm's hipcc in PATH cannot link (no libamdhip64.so symlink): use the /opt one.
HIPCC=/opt/rocm/core-10.1/bin/hipcc ./build.sh /root/w208/bin            # ~25 s, 29 binaries
A="--mm 16,24,32,40,48,64"; L=/root/card0.lock
./run.sh /root/w208/bin out/p1.jsonl $L $A --check 0            # a timing pass (GEMV M 1,2,4,8 + GEMM 256), ~20 min; x3
./run.sh /root/w208/bin out/d1.jsonl $L $A --M 0 --check 0      # a decode-pass timing pass, ~8 min
./run.sh /root/w208/bin out/c1.jsonl $L --mm 8,16,24,32,40,48,64 --check 1 --reps 3 --batch 16 --warm 0.1 --warm-cfg 0.02   # exactness
python3 report.py out/p?.jsonl out/d?.jsonl --check out/c1.jsonl --asm asm.jsonl
python3 asmcount.py asm /opt/rocm/core-10.1/lib/llvm/bin/clang++ cb208.hip 32 1 0 >> asm.jsonl   # MM M A4
# box2 card 2 (R9700): HIPCC=/opt/rocm/bin/hipcc, ROCR_VISIBLE_DEVICES=2, lock /data/q27/locks/card2.lock
# one binary by hand:
flock /root/card0.lock bin/cb208_Mad --shapes gate_up --mm 16,24 --M 1,8,256 [--check 0] [--out f.jsonl]
CANDS="Mad Cptc5Lds" ./run.sh ...                               # a subset
```

`--mm` is 16 K (8 = K 0.5 … 64 = K 4), `--M` the token counts (0 = the decode pass, 256 = the WMMA GEMM), `--shapes`
from `gate_up,down,tiny`. The binary refuses any device that is not gfx1201 (the iGPU). Exit code 2 = an exactness
failure. `run.sh` takes the card lock around each binary, so other users of the card get in between candidates.
Exactness runs hold the card while the host oracle computes (slow, ~1 min a candidate): keep them apart from timing
passes. `--warm-cfg` below 0.1 s skips the long warm-up (exactness-only runs; such times are not used).

### What is in `results.jsonl` (dad1) — `merge.py`

Each line carries `"pass"`: `p1..p3` three timing passes, `d1, d2` decode passes, `c1` the exactness pass (its times
are ignored by `report.py`). The passes were not one clean sweep: the CPTC multiply fix (below) came after the first
passes, so p1 / p2 of the candidates the fix did not touch (raw, base, mad, mad4, qtip_3inst, state tables) are from
the first build and everything else from the rebuilt binaries; the kernels of the untouched candidates are the same
code in both builds. The first build's lines of the fixed candidates are `results-v1-mul_lo.jsonl`. The L 6 / L 8
state-table probes were added last. The exactness pass ran a final build that differs from the timed one only in the
warm-up rule.

## Stream and trellis (the fractional-K requirement)

The exl3i ring, bit for bit (`plugin/exl3import/common/exl3i_decode.h` on `qc-exl3-v3`, = `v3/e3/viterbi.c`
`viterbi_tiles_pat` with `trellis.py` `pattern`): a tile is 256 positions at K = MM / 16, MM / 2 u32 words, one MSB-first
ring of 16 MM bits. Position i takes D(i) = ⌊(i%16 + 1) MM / 16⌋ − ⌊(i%16) MM / 16⌋ new bits; its state is the 16 ring
bits ending at S(i) = D(0) + … + D(i), circular (tail-biting); a 0-bit step reads the codebook at
`state ^ salt(i)`, salt(i) = 0x9E37 (i%16 + 1) mod 2^16. Every candidate is a function of that 16-bit state, so all of
them run every pattern; the shifts of a pattern are compile-time constants of the kernel (template parameter MM), as in
production's rate plans. K 0.5 (0-bit steps, the salt) is run and checked too.

Only the scan is this bench's own: position i = tile row i / 16 (output), column i % 16 (input), so a lane owns whole
rows and needs no lane exchange (a quantizer would encode in this scan; production's `tensor_core_perm` scan costs the
same decode ALU, it changes which lane holds which run).

## Kernels

- **GEMV, M = 1, 2, 4, 8:** a lane owns one tile row: MM / 32 + 1 … + 2 dword loads (the row's bits and 16 bits of
  look-back), one 64-bit shift a word to align, then a bit-field extract a state, the codebook, and a pack of the 16
  grid indices into 4 dwords (int8) or 2 dwords (int4). Activations: int8 4 a dword or int4 8 a dword, the M tokens
  interleaved; one `v_dot4_i32_iu8` / `v_dot8_i32_iu4` a packed dword a token. Sums are int32 over the whole row
  (per-token scale; no 128-column float groups, which would add the same work to every candidate). A wave takes two
  tiles a step; KW waves split the columns and reduce through LDS; KW ∈ {2, 4, 8, 16} and one or two tile rows a block
  are tuned per configuration and candidate (the timed choice is in the result line).
- **GEMM, M = 256:** a lane owns half a tile row (the gfx12 WMMA A fragment), a wave decodes one tile and multiplies it
  into 16 or 8 token tiles (`wmma_i32_16x16x16_iu8` / `_iu4`), activation fragments straight from global memory.
  This is a straightforward GEMM (no LDS staging of activations, the decode is repeated when the tokens are split
  over two waves): its µs are not a tuned prefill number, the factor over its own baseline is the point.
- Weights are decoded as unsigned grid indices n; the epilogue subtracts bias · Σx (one precomputed int a token).
- **`base`** (the factor's denominator): the same kernel, the "codebook" is the state's low 8 (4) bits: loads, align,
  one bit-field extract a weight, pack, dot. The paper's INT4-baseline analogue. **`raw`**: no per-weight work at all
  (the aligned words are used as the packed weights): what loads + dots alone cost.
- **Decode pass (`--M 0`):** the same GEMV kernel writing the packed int8 / int4 weights to memory instead of
  multiplying: what a two-pass prefill (decode once, then an integer GEMM that does not depend on the codebook) pays
  for a codebook. The output buffer is reused by every launch (as a prefill scratch buffer would be).
- Context lines: `read` = a pure 16-byte-vector read of the same weight bytes (GB/s, best grid), `empty` = an empty
  launch of the GEMV's grid.

## Candidates

n = the unsigned grid index a kernel multiplies; value = n − bias. s = the 16-bit state. Constants in `cb208.hip`.

| name (binary) | definition | type | table |
|---|---|---|---|
| mad (`Mad`) | n = bytesum((s·0xD0474B + 0x2545F491) & 0x3F3F3F3F) + 2, bias 128 (production) | int8 | – |
| mad [fold] (`MadF`) | the same value, decode folded into the dot (`madgemv` #177): Σ sudot4(h, x·0x01010101); one dot a weight a token, activations 4× the bytes | int8 | – |
| **mad4** (`Mad4`) | **proposal:** h = (s·0xD0474B + 0x2545F491) mod 2^32; n = ((h>>28)&7) + ((h>>24)&7) + ((h>>20)&1) = `udot8(h & 0x77100000, 0x11100000)`, value n − 8 ∈ [−8, 7], 16 levels (3 + 3 + 1-bit fields of the hash's top bits; near-triangular, sd 3.28 levels). Same three instructions as mad (mad24, and, dot8) | int4 | – |
| mad4 [fold] (`Mad4F`) | w4a4 only: Σ sudot8(h & mask, x·0x11100000) | int4 | – |
| mul1_i8 (`Mul1I8`) | trellis.py `mul1_i8`: bytesum(s·0x83DCD12D) standardised, rounded to ±127 = 3.5 sd; device: n = (bs·32175 + 434171) >> 17 (equal for every byte sum; checked on all states) | int8 | – |
| int4grid (`Int4Grid`) | trellis.py `int4grid`: the same byte sum on 16 levels; device n = clamp((bs·1279 − 127977) >> 16, 0, 15), value n − 7.5 | int4 | – |
| qtip_1mad (`Qtip1Mad`) | (bytesum(s·34038481 + 76625530) − 510) / 147.8005 rounded to the int8 grid; device (bs·8045 + 107737) >> 15 | int8 | – |
| qtip_3inst (`Qtip3Inst`) | x = s·89226354 + 64248484; fp16(lo16 ^ m) + fp16(hi16 ^ m) in half arithmetic (mask 0x8FFF8FFF, m 0x3B60), × 127 / (3.5 sd), rint. Straightforward float path (2 cvt, add_f16, mul, rndne, cvt) | int8 | – |
| lc1 (`Lc1Lds`, `Lc1Glb`) | one multiply: n = T[(a1·s mod 2^16) >> 10], a1 = 0xA981 | int8 | 64 B |
| cptc5 (`Cptc5Lds`, `Cptc5Glb`, `Cptc5Reg`) | i = ((a0·s mod 2^16) >> 11) + ((a1·s mod 2^16) >> 11), n = T[i], 63 entries in full; unified pair (0x9282, 0xA981) | int8 | 63 B |
| cptc6 (`Cptc6Lds`, `Cptc6Glb`, `Cptc6Eq3`) | Algorithm 1: i with >> 10 (0..126), mirror fold (i < 64 ? T[i] : −T[126 − i]), 64-entry table | int8 | 64 B |
| cptc6full (`Cptc6FLds`, `Cptc6FGlb`) | the same values, the 127 entries stored out (no fold) — an extra this bench adds | int8 | 127 B |
| cptcst10 / cptcst12 (`St10*`, `St12*`) | n = T[s & (2^L − 1)], L = 10 / 12 (a depth-L trellis) | int8 | 1 KB / 4 KB |
| cptcst6 / cptcst8 (`St6*`, `St8*`) | the same decode with a 64 B / 256 B table: locality probes of this bench, not candidates | int8 | 64 B / 256 B |
| cptcmin (`CptcMin`) | Q = ⌈K⌉ + 1: n = i (the 2^(Q+1) − 1 levels are the index itself, no table), bias 2^Q − 1; int4 for K ≤ 2 | int4 (K ≤ 2) / int8 | – |

Placements: `lds` = the table copied into LDS at block start (byte loads), `glb` = read from global memory (L1/L2
resident), `reg` (cptc5) = the 64 bytes in 16 registers, 8 `v_perm_b32` (8 entries each) + a 3-level byte select for
4 weights at once. `lds+eq3` (cptc6) = the paper's Eq. (3): inside a run the two products follow
x_c = (x_{c−1} << D) + a·code_c (a D-bit × 16 multiply and a shift-add instead of a 16 × 16 multiply).
All table candidates are within the 8 KB a tensor budget (the largest, cptcst12, is 4 KB).

**CPTC multiplies (a codegen trap worth knowing for the plugin):** the products are 16 × 16-bit and should be
`v_mul_u32_u24`. Written naively they are not: (1) `__builtin_amdgcn_ubfe(s * a, 10, 6)` is rewritten by LLVM to
`(s * (a << 16)) >> 26`, a full `v_mul_lo_u32`; (2) with a plain `(p >> 10) & 63` LLVM still drops the state's 16-bit
mask (the index needs only the product's low 16 bits), so the operand is a full 32-bit value and half the products of
a run are again `v_mul_lo_u32`. The bench therefore takes a CPTC candidate's states with a `v_bfe_u32` whose offset is
a constant plus a run-time zero (`ST8` in `states()`): the bit-field node survives with 16 known bits and all products
select `v_mul_u32_u24`. The first three passes on dad1 ran form (1) (`results-v1-mul_lo.jsonl`, kept as the measured
cost of the trap); every number in the tables is the fixed form. mul1 / qtip hashes need a full multiply by
definition (32-bit constants; mad's 24-bit constant was chosen to avoid exactly this); their byte-sum → grid map is a
24-bit mad after an explicit `& 1023`.

Tables are the paper's Eq. (2) start tables for the unified pair at INT8 scale 32 (no Lloyd rounds): decode time does
not depend on the table contents; the quality axis is the lead's. lc1's multiplier and the state tables' cell order
are placeholders for the same reason.

## Exactness

For every candidate (all 29 binaries), K ∈ {0.5, 1, 1.5, 2, 2.5, 3, 4}, both shapes and modes: (1) the device's decoded weights (a dump
variant of the same GEMV kernel) against a host oracle (states bit by bit from the ring; the codebook from its float /
integer definition, not from the device code), every weight; (2) the int32 outputs of every timed GEMV configuration
against the host product of the oracle weights, every row; (3) the GEMM's outputs on every 53rd row. All integer,
all exact (`"bad":0` in each line; `"test":"decode"` lines are (1)).
Final pass on dad1 (`"pass":"c1"`): 442 decode tests (7.82 G weights), 1,768 GEMV and 442 GEMM configurations,
0 failures (`raw` has no codebook and is not checked). box2: the same on the tiny shape (48 × 96).

## Timing protocol

Wall clock around `batch` back-to-back launches closed by a device synchronize (GEMV 128 launches, GEMM 4); 31
repetitions; median (also min and p90 in the line). Weights rotate over enough copies (≤ 64, ≥ 256 MB where 64 allow)
that every launch reads cold weights. 0.4 s of load before anything, 0.15 s before each configuration (the card drops
to idle clocks during host work; without this the first configuration after a pause read up to 1.6× slow), then the
tuning passes, then the timed run. One kernel at a time under the card lock. Clocks and power cap as found.
The report takes, per configuration, the median over the passes (three for GEMV / GEMM; pass-to-pass spread of a
configuration: median 0.4%, max 9% GEMV, 12% GEMM).
