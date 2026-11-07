// beamL.c -- w71 (#113): bitshift trellis with an L-bit state window (L = 16..20) on 16x16 tiles (256 values),
// integer K bits a step, searched by full Viterbi (M = 0) or by a beam (M-algorithm, M survivors), with a free start
// (tb = 0, settle.py's default path) or exllamav3's tail-biting heuristic (tb = 1).
//
// State after step i: s_i = ((s_{i-1} << K) | x_i) & (2^L - 1); the value emitted at step i is cb[s_i] (cb: 2^L
// floats; a computed code such as mul1 is a function of the state, so the table is only the encoder's cache).
// Tail-biting (EXL3's format constraint; the stream of 256 K-bit symbols is circular): (s_0 >> K) == (s_255 & (2^(L-K)
// - 1)). exllamav3's heuristic (quantize_tiles_kernel.cuh; viterbi.c viterbi_tiles_tb with ntry 1): a free run on
// the tile rolled by 128 gives the wrap group c (the state at original position 0, >> K), then one run with s_0 in
// {(c << K) | x} and the end state forced onto c. The beam forces the end by fixing the bits of the last symbols that
// fall into the end state's low L - K bits (exactly equivalent for Viterbi, where viterbi.c restricts the end states).
//
// Full Viterbi (M = 0): viterbi.c's loops with 16 replaced by L (same arithmetic order and tie rules: lowest
// predecessor top bits j on ties, lowest end state), so L = 16 reproduces viterbi_tiles_pat (free) and
// viterbi_tiles_tb ntry 1 bit for bit.
// Beam (M > 0): survivors are distinct states. Survivors sharing their low L - K bits (group g) have identical
// children ((g << K) | x), so each group keeps only its cheapest member (ties: lowest state, as Viterbi's lowest j);
// the deduped parents expand to their 2^K children (distinct across groups); the M cheapest children survive (ties
// at the cut: lowest candidate index; the cut is found by value histograms, exact). M >= 2^L keeps every state: exact Viterbi.
//   gcc -O3 -march=native -fopenmp -shared -fPIC beamL.c -o libbeamL.so
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// ---- full Viterbi, one run (viterbi.c tb_run with an L-bit window). roll: the tile is read rotated by roll;
// c < 0: free start / free end; else start states with (s >> K) == c, end states with (s & (NG - 1)) == c.
static float vit_run(const float* xt, const float* cb, int L, int K, int roll, int c, float* cost, float* ncost,
                     float* gmin, uint8_t* bj, uint8_t* bp, uint32_t* st) {
  const int NS = 1 << L, SH = L - K, NG = NS >> K, NJ = 1 << K;
  { const float x0 = xt[roll & 255];
    for (int s = 0; s < NS; s++) { const float d = x0 - cb[s]; cost[s] = (c < 0 || (s >> K) == c) ? d * d : FLT_MAX; } }
  for (int i = 1; i < 256; i++) {
    const float xi = xt[(i + roll) & 255];
    for (int g = 0; g < NG; g++) { gmin[g] = cost[g]; bj[g] = 0; }
    for (int j = 1; j < NJ; j++) { const float* cj = cost + ((size_t)j << SH);
      for (int g = 0; g < NG; g++) { const int lt = cj[g] < gmin[g]; gmin[g] = lt ? cj[g] : gmin[g]; bj[g] = lt ? (uint8_t)j : bj[g]; } }
    memcpy(bp + (size_t)i * NG, bj, NG);
    for (int s = 0; s < NS; s++) { const float d = xi - cb[s]; ncost[s] = gmin[s >> K] + d * d; }
    float* t = cost; cost = ncost; ncost = t;
  }
  int sb = -1; float m = FLT_MAX;
  for (int s = 0; s < NS; s++) if ((c < 0 || (s & (NG - 1)) == c) && cost[s] < m) { m = cost[s]; sb = s; }
  if (sb < 0) return FLT_MAX;
  st[255] = (uint32_t)sb;
  for (int i = 255; i > 0; i--) { const uint32_t g = st[i] >> K; st[i - 1] = ((uint32_t)bp[(size_t)i * NG + g] << SH) | g; }
  return m;
}

// ---- beam (M-algorithm) ------------------------------------------------------------------------------------------
typedef struct {
  int L, K, M;
  uint32_t *st0, *st1;    // survivors' states (current, next)
  float *co0, *co1;       // their path costs
  uint32_t *hst;          // [256][M] survivor states per step
  uint32_t *hpar;         // [256][M] survivor's parent index in the previous step's list
  int *gidx;              // [2^(L-K)] group -> deduped parent (-1 = none)
  int *uniq;              // deduped parents (indices into the current list)
  float *cc;              // candidate costs [M * 2^K] (step 0: 2^L)
  float *sc;              // scratch values for selection
  int *si;                // scratch candidate indices
  int *sel;               // selected candidate indices
} beam_t;

static void* xmalloc(size_t n) { void* p = malloc(n ? n : 1); if (!p) abort(); return p; }

static beam_t* beam_new(int L, int K, int M) {
  beam_t* b = xmalloc(sizeof *b); b->L = L; b->K = K; b->M = M;
  const size_t NJ = (size_t)1 << K, NC = (size_t)M * NJ > ((size_t)1 << L) ? (size_t)M * NJ : ((size_t)1 << L);
  b->st0 = xmalloc(4 * (size_t)M); b->st1 = xmalloc(4 * (size_t)M);
  b->co0 = xmalloc(4 * (size_t)M); b->co1 = xmalloc(4 * (size_t)M);
  b->hst = xmalloc(4 * 256 * (size_t)M); b->hpar = xmalloc(4 * 256 * (size_t)M);
  b->gidx = xmalloc(sizeof(int) * ((size_t)1 << (L - K))); for (size_t g = 0; g < ((size_t)1 << (L - K)); g++) b->gidx[g] = -1;
  b->uniq = xmalloc(sizeof(int) * (size_t)M);
  b->cc = xmalloc(4 * NC); b->sc = xmalloc(4 * NC); b->sel = xmalloc(sizeof(int) * (size_t)M);
  return b;
}
static void beam_free(beam_t* b) {
  free(b->st0); free(b->st1); free(b->co0); free(b->co1); free(b->hst); free(b->hpar); free(b->gidx); free(b->uniq);
  free(b->cc); free(b->sc); free(b->sel); free(b);
}

// the k-th smallest (0-based) of v[0..n): value histograms, each level on the bin holding the k-th, until the bin
// is small (then an insertion sort). sc / si are scratch of n floats. v is not modified.
#define NB 2048
static float kth_hist(const float* v, int n, int k, float* sc) {
  static __thread int cnt[NB];
  const float* w = v; int m = n;
  for (int lev = 0; ; lev++) {
    float lo = w[0], hi = w[0];
    for (int q = 1; q < m; q++) { lo = w[q] < lo ? w[q] : lo; hi = w[q] > hi ? w[q] : hi; }
    if (lo == hi) return lo;
    if (m <= 48) {                                     // insertion sort into sc (w may be sc: copy first)
      float t[48]; memcpy(t, w, 4 * (size_t)m);
      for (int i = 1; i < m; i++) { const float x = t[i]; int j = i - 1; while (j >= 0 && t[j] > x) { t[j + 1] = t[j]; j--; } t[j + 1] = x; }
      return t[k];
    }
    const float inv = (float)NB / (hi - lo);
    memset(cnt, 0, sizeof cnt);
    for (int q = 0; q < m; q++) { int bq = (int)((w[q] - lo) * inv); bq = bq < NB ? bq : NB - 1; cnt[bq]++; }
    int b = 0, cum = 0; while (cum + cnt[b] <= k) { cum += cnt[b]; b++; }
    k -= cum;
    int m2 = 0;                                        // compact bin b into sc (in place when w == sc: m2 <= q)
    for (int q = 0; q < m; q++) { const float x = w[q]; int bq = (int)((x - lo) * inv); bq = bq < NB ? bq : NB - 1; sc[m2] = x; m2 += bq == b; }
    w = sc; m = m2;
  }
}

// select the M cheapest of the n candidate costs cc (ties at the cut: lowest index), indices ascending into sel
static int select_m(beam_t* b, int n, int M) {
  const float* cc = b->cc; int* sel = b->sel;
  if (n <= M) { for (int k = 0; k < n; k++) sel[k] = k; return n; }
  const float v = kth_hist(cc, n, M - 1, b->sc);
  int nl = 0; for (int q = 0; q < n; q++) nl += cc[q] < v;
  int ne = M - nl, m = 0;
  for (int q = 0; q < n; q++) { const float y = cc[q]; const int take = (y < v) | ((y == v) & (ne > 0)); ne -= (y == v) & take; sel[m] = q; m += take; }
  return m;
}

// one beam run: as vit_run (roll, c), returns the cost and the states st[256]
static float beam_run(beam_t* b, const float* xt, const float* cb, int roll, int c, uint32_t* st) {
  const int L = b->L, K = b->K, M = b->M, NJ = 1 << K, SH = L - K;
  const uint32_t LM = (1u << L) - 1, GM = (1u << SH) - 1;
  int fm[256], fv[256];                                 // forced bits of each symbol (end constraint)
  for (int i = 0; i < 256; i++) fm[i] = fv[i] = 0;
  if (c >= 0) for (int bb = 0; bb < SH; bb++) { const int i = 255 - bb / K, t = bb % K; fm[i] |= 1 << t; fv[i] |= ((c >> bb) & 1) << t; }
  float* cc = b->cc; int n;
  // step 0
  { const float x0 = xt[roll & 255];
    if (c < 0) { n = 1 << L; for (int s = 0; s < n; s++) { const float d = x0 - cb[s]; cc[s] = d * d; } }
    else { n = 0; for (int x = 0; x < NJ; x++) if ((x & fm[0]) == fv[0]) { const uint32_t s = ((uint32_t)c << K) | x; const float d = x0 - cb[s]; cc[n++] = d * d; } }
    const int m = select_m(b, n, M);
    int q = 0;
    if (c < 0) for (int k = 0; k < m; k++) { b->st0[k] = (uint32_t)b->sel[k]; b->co0[k] = cc[b->sel[k]]; }
    else { int xs[256], nx = 0; for (int x = 0; x < NJ; x++) if ((x & fm[0]) == fv[0]) xs[nx++] = x;
           for (int k = 0; k < m; k++) { b->st0[k] = ((uint32_t)c << K) | xs[b->sel[k]]; b->co0[k] = cc[b->sel[k]]; } }
    q = m;
    memcpy(b->hst, b->st0, 4 * (size_t)q);
    b->hpar[0] = 0; n = q;
  }
  int ncur = n;
  for (int i = 1; i < 256; i++) {
    const float xi = xt[(i + roll) & 255];
    // dedupe by group
    int nu = 0;
    for (int j = 0; j < ncur; j++) {
      const uint32_t g = b->st0[j] & GM; const int o = b->gidx[g];
      if (o < 0) { b->gidx[g] = j; b->uniq[nu++] = j; }
      else if (b->co0[j] < b->co0[o] || (b->co0[j] == b->co0[o] && b->st0[j] < b->st0[o])) b->gidx[g] = j;
    }
    for (int u = 0; u < nu; u++) { const uint32_t g = b->st0[b->uniq[u]] & GM; b->uniq[u] = b->gidx[g]; }
    for (int u = 0; u < nu; u++) b->gidx[b->st0[b->uniq[u]] & GM] = -1;
    // expand
    int xs[256], nx = 0; for (int x = 0; x < NJ; x++) if ((x & fm[i]) == fv[i]) xs[nx++] = x;
    for (int u = 0; u < nu; u++) {
      const int p = b->uniq[u]; const float pc = b->co0[p]; const uint32_t base = ((b->st0[p] & GM) << K);
      float* cu = cc + (size_t)u * nx;
      if (nx == NJ) { const float* cbb = cb + base; for (int x = 0; x < NJ; x++) { const float d = xi - cbb[x]; cu[x] = pc + d * d; } }
      else for (int k = 0; k < nx; k++) { const float d = xi - cb[base | xs[k]]; cu[k] = pc + d * d; }
    }
    n = nu * nx;
    const int m = select_m(b, n, M);
    uint32_t* hs = b->hst + (size_t)i * M; uint32_t* hp = b->hpar + (size_t)i * M;
    for (int k = 0; k < m; k++) {
      const int q = b->sel[k], u = q / nx, p = b->uniq[u];
      const uint32_t s = (((b->st0[p] & GM) << K) | (uint32_t)xs[q - u * nx]) & LM;
      b->st1[k] = s; b->co1[k] = cc[q]; hs[k] = s; hp[k] = (uint32_t)p;
    }
    ncur = m;
    uint32_t* t = b->st0; b->st0 = b->st1; b->st1 = t; float* tc = b->co0; b->co0 = b->co1; b->co1 = tc;
  }
  int jb = -1; float mm = FLT_MAX;
  for (int j = 0; j < ncur; j++) if (b->co0[j] < mm || (b->co0[j] == mm && jb >= 0 && b->st0[j] < b->st0[jb])) { mm = b->co0[j]; jb = j; }
  if (jb < 0) return FLT_MAX;
  for (int i = 255; i >= 0; i--) { st[i] = b->hst[(size_t)i * M + jb]; if (i) jb = (int)b->hpar[(size_t)i * M + jb]; }
  return mm;
}

// ---- driver: x [ntiles][256] targets -> out [ntiles][256] values (idx: states, may be NULL), cost: per-tile SSE of
// the search (may be NULL). M = 0: full Viterbi; tb = 0: free start (one run), 1: exllamav3's tail-biting heuristic.
void trellis_tiles(const float* x, float* out, uint32_t* idx, float* cost, int ntiles, const float* cb, int L, int K,
                   int M, int tb) {
#pragma omp parallel
  {
    const int NS = 1 << L, NG = NS >> K;
    float *vc = NULL, *vn = NULL, *vg = NULL; uint8_t *vbj = NULL, *vbp = NULL; beam_t* bm = NULL;
    if (M == 0) { vc = xmalloc(4 * (size_t)NS); vn = xmalloc(4 * (size_t)NS); vg = xmalloc(4 * (size_t)NG);
                  vbj = xmalloc(NG); vbp = xmalloc((size_t)256 * NG); }
    else bm = beam_new(L, K, M);
    uint32_t st[256];
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256; float m;
#define RUN(roll, c) (M == 0 ? vit_run(xt, cb, L, K, roll, c, vc, vn, vg, vbj, vbp, st) : beam_run(bm, xt, cb, roll, c, st))
      if (!tb) m = RUN(0, -1);
      else { RUN(128, -1); const int c = (int)(st[128] >> K); m = RUN(0, c); }
#undef RUN
      for (int i = 0; i < 256; i++) { out[(size_t)t * 256 + i] = cb[st[i]]; if (idx) idx[(size_t)t * 256 + i] = st[i]; }
      if (cost) cost[t] = m;
    }
    if (M == 0) { free(vc); free(vn); free(vg); free(vbj); free(vbp); } else beam_free(bm);
  }
}
