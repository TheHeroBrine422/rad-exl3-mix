// beamS.c -- #177 (w139): seeded L-bit search on 16x16 tiles, on top of w71's beamL.c (#113; included unchanged).
// Lead notes 11-13 (1:45 / 2:00 PM CST): any K-bit symbol stream is a valid path at every L (the stream is circular,
// state s_i = the last ceil(L / K) symbols up to i, masked to L bits), so an exact lower-L path can seed the L search.
//   mode 1  seeded beam: exact tail-biting Viterbi at Ls (table cbs) -> its symbols read as L-bit states -> beam at L
//           (M survivors) constrained to the seed's wrap group, the seed's state forced into the survivors at every
//           step: the result never costs more than the seed under the L table.
//   mode 2  segment refine: the seed path; free n = NB / K consecutive symbols (circular), exact DP over them with the
//           other symbols fixed (every position whose window holds a free symbol is costed), keep it if cheaper;
//           segments at stride max(1, n / 2) around the tile, passes until one gains nothing (cap 8).
//   mode 3  seeded beam, then segment refine.
//   mode 4  (no seed: a lower-L path is ~random under the L table of a computed code, 11-13 test) plain beam at L
//           (beamL.c's beam_run, tail-biting heuristic) as the start; 6 = that beam, then segment refine.
// Arguments: M = beam survivors (modes 1 / 3), NB = free bits a segment (modes 2 / 3).
//   gcc -O3 -march=native -fopenmp -shared -fPIC beamS.c -o libbeamS.so   (beamL.c next to it)
#include "beamL.c"

static void states_from_syms(const uint8_t* sym, int L, int K, uint32_t* st) {
  const int W = (L + K - 1) / K; const uint32_t LM = (1u << L) - 1;
  for (int i = 0; i < 256; i++) { uint32_t s = 0; for (int t = W - 1; t >= 0; t--) s = (s << K) | sym[(i - t) & 255]; st[i] = s & LM; }
}
static float path_cost(const float* xt, const float* cb, const uint32_t* st) {
  float c = 0; for (int i = 0; i < 256; i++) { const float d = xt[i] - cb[st[i]]; c += d * d; } return c;
}

// the beam of beamL.c with the seed's states ss[256] forced into the survivors (c = (ss[0] >> K))
static float beam_run_seeded(beam_t* b, const float* xt, const float* cb, const uint32_t* ss, uint32_t* st) {
  const int L = b->L, K = b->K, M = b->M, NJ = 1 << K, SH = L - K;
  const uint32_t LM = (1u << L) - 1, GM = (1u << SH) - 1; const int c = (int)(ss[0] >> K);
  int fm[256], fv[256];
  for (int i = 0; i < 256; i++) fm[i] = fv[i] = 0;
  for (int bb = 0; bb < SH; bb++) { const int i = 255 - bb / K, t = bb % K; fm[i] |= 1 << t; fv[i] |= ((c >> bb) & 1) << t; }
  float* cc = b->cc; int n;
  { const float x0 = xt[0]; int xs[256], nx = 0, qs = -1;
    for (int x = 0; x < NJ; x++) if ((x & fm[0]) == fv[0]) { if ((((uint32_t)c << K) | x) == ss[0]) qs = nx; xs[nx++] = x; }
    for (int k = 0; k < nx; k++) { const float d = x0 - cb[((uint32_t)c << K) | xs[k]]; cc[k] = d * d; }
    n = nx; int m = select_m(b, n, M);
    int hit = 0; for (int k = 0; k < m; k++) hit |= b->sel[k] == qs;
    if (!hit) { int w = 0; for (int k = 1; k < m; k++) if (cc[b->sel[k]] > cc[b->sel[w]]) w = k; b->sel[w] = qs; }
    for (int k = 0; k < m; k++) { b->st0[k] = ((uint32_t)c << K) | xs[b->sel[k]]; b->co0[k] = cc[b->sel[k]]; }
    memcpy(b->hst, b->st0, 4 * (size_t)m); b->hpar[0] = 0; n = m;
  }
  int ncur = n;
  for (int i = 1; i < 256; i++) {
    const float xi = xt[i];
    int nu = 0;
    for (int j = 0; j < ncur; j++) {
      const uint32_t g = b->st0[j] & GM; const int o = b->gidx[g];
      if (o < 0) { b->gidx[g] = j; b->uniq[nu++] = j; }
      else if (b->co0[j] < b->co0[o] || (b->co0[j] == b->co0[o] && b->st0[j] < b->st0[o])) b->gidx[g] = j;
    }
    for (int u = 0; u < nu; u++) { const uint32_t g = b->st0[b->uniq[u]] & GM; b->uniq[u] = b->gidx[g]; }
    for (int u = 0; u < nu; u++) b->gidx[b->st0[b->uniq[u]] & GM] = -1;
    int xs[256], nx = 0, ks = -1;
    for (int x = 0; x < NJ; x++) if ((x & fm[i]) == fv[i]) { if ((uint32_t)x == (ss[i] & (NJ - 1))) ks = nx; xs[nx++] = x; }
    int us = -1; const uint32_t gs = ss[i] >> K;
    for (int u = 0; u < nu; u++) {
      const int p = b->uniq[u]; const float pc = b->co0[p]; const uint32_t base = ((b->st0[p] & GM) << K);
      if ((b->st0[p] & GM) == gs) us = u;
      float* cu = cc + (size_t)u * nx;
      for (int k = 0; k < nx; k++) { const float d = xi - cb[base | xs[k]]; cu[k] = pc + d * d; }
    }
    if (us < 0 || ks < 0) abort();                       // the seed's parent group must survive (induction)
    n = nu * nx;
    const int m = select_m(b, n, M), qs = us * nx + ks;
    int hit = 0; for (int k = 0; k < m; k++) hit |= b->sel[k] == qs;
    if (!hit) { int w = 0; for (int k = 1; k < m; k++) if (cc[b->sel[k]] > cc[b->sel[w]]) w = k; b->sel[w] = qs; }
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
  for (int i = 255; i >= 0; i--) { st[i] = b->hst[(size_t)i * M + jb]; if (i) jb = (int)b->hpar[(size_t)i * M + jb]; }
  return mm;
}

// one segment: symbols a .. a+n-1 free (circular), exact DP; positions a .. a+n+W-2 costed. DP index at position p =
// the free symbols inside p's window, most recent in the low bits. Returns the gain (>= 0) and rewrites sym on a gain.
typedef struct { float *c0, *c1; uint8_t* bp; } seg_t;
static float segment(const float* xt, const float* cb, int L, int K, uint8_t* sym, int a, int n, seg_t* w) {
  const int W = (L + K - 1) / K, NP = n + W - 1, KM = (1 << K) - 1; const uint32_t LM = (1u << L) - 1;
  uint8_t sv[256 + 32];
  float old = 0;
  for (int j = 0; j < NP; j++) { const int p = (a + j) & 255; uint32_t s = 0;
    for (int t = W - 1; t >= 0; t--) s = (s << K) | sym[(p - t) & 255];
    const float d = xt[p] - cb[s & LM]; old += d * d; }
  // f_j = number of free symbols in window j; window j holds free symbols max(0, j-W+1) .. min(j, n-1)
  float *c0 = w->c0, *c1 = w->c1; int f0 = 0; size_t bpo = 0; int fs[256 + 32]; size_t bpoff[256 + 32];
  c0[0] = 0;
  for (int j = 0; j < NP; j++) {
    const int lo = j - W + 1 > 0 ? j - W + 1 : 0, hi = j < n - 1 ? j : n - 1, f = hi - lo + 1;
    const int enter = j < n, drop = f0 + enter - f;     // drop: 0 or 1 (oldest free symbol leaves)
    const int p = (a + j) & 255; const float xp = xt[p];
    // fixed part of the state: symbols in the window that are not free
    uint32_t fixed = 0;
    for (int t = W - 1; t >= 0; t--) { const int jj = j - t; const int isfree = jj >= 0 && jj < n;
      fixed = (fixed << K) | (isfree ? 0 : sym[(p - t) & 255]); }
    // free symbol jj sits at shift (j - jj) * K
    const int NI = 1 << (K * f);
    fs[j] = f; bpoff[j] = bpo;
    for (int v = 0; v < NI; v++) {
      // v: free symbols lo..hi, symbol hi in the low bits
      float best; int bd = 0;
      if (!drop) { const int pv = enter ? v >> K : v; best = c0[pv]; }
      else { const int pv0 = (enter ? v >> K : v); best = FLT_MAX;
        // previous index: free symbols lo-1..hi_prev, oldest (lo-1) in the top bits of width K at shift K*(f0-1)
        for (int d = 0; d <= KM; d++) { const float cv = c0[pv0 | (d << (K * (f0 - 1)))]; if (cv < best) { best = cv; bd = d; } } }
      uint32_t s = fixed;
      for (int q = 0; q < f; q++) s |= (uint32_t)((v >> (K * q)) & KM) << (K * (j - (hi - q)));
      const float d = xp - cb[s & LM];
      c1[v] = best + d * d;
      if (drop) w->bp[bpo + v] = (uint8_t)bd;
    }
    if (drop) bpo += NI;
    float* t = c0; c0 = c1; c1 = t; f0 = f;
  }
  // end: window NP-1 holds only free symbol n-1 (f0 = 1 when W >= 1)
  int vb = 0; float mb = FLT_MAX; const int NI = 1 << (K * f0);
  for (int v = 0; v < NI; v++) if (c0[v] < mb) { mb = c0[v]; vb = v; }
  if (!(mb < old - 1e-7f * (old > 1 ? old : 1))) return 0;
  // backtrack: at the end the index holds symbols lo..n-1; walk back through drops
  int v = vb;
  for (int j = NP - 1; j >= 0; j--) {
    const int lo = j - W + 1 > 0 ? j - W + 1 : 0, hi = j < n - 1 ? j : n - 1, f = fs[j];
    for (int q = 0; q < f; q++) sv[hi - q] = (uint8_t)((v >> (K * q)) & KM);
    if (j == 0) break;
    const int enter = j < n, fp = fs[j - 1], drop = fp + enter - f;
    int pv = enter ? v >> K : v;
    if (drop) pv |= w->bp[bpoff[j] + v] << (K * (fp - 1));
    (void)lo; v = pv;
  }
  for (int jj = 0; jj < n; jj++) sym[(a + jj) & 255] = sv[jj];
  return old - mb;
}

static float refine(const float* xt, const float* cb, int L, int K, uint8_t* sym, int NBb, seg_t* w) {
  int n = NBb / K; if (n < 1) n = 1; const int stride = n / 2 > 0 ? n / 2 : 1;
  for (int pass = 0; pass < 8; pass++) {
    float g = 0; for (int a = 0; a < 256; a += stride) g += segment(xt, cb, L, K, sym, a, n, w);
    if (!(g > 0)) break;
  }
  uint32_t st[256]; states_from_syms(sym, L, K, st); return path_cost(xt, cb, st);
}

// x [ntiles][256] -> out values, cost (may be NULL). cbs: the Ls table of the seed (exact tail-biting Viterbi).
void trellis_tiles_seeded(const float* x, float* out, float* cost, float* seedcost, int ntiles, const float* cb, int L,
                          const float* cbs, int Ls, int K, int mode, int M, int NBb) {
#pragma omp parallel
  {
    const int NS = 1 << ((mode & 4) ? 1 : Ls), NG = NS >> ((mode & 4) ? 1 : K);
    float *vc = xmalloc(4 * (size_t)NS), *vn = xmalloc(4 * (size_t)NS), *vg = xmalloc(4 * (size_t)NG);
    uint8_t *vbj = xmalloc(NG), *vbp = xmalloc((size_t)256 * NG);
    beam_t* bm = (mode & 5) ? beam_new(L, K, M) : NULL;
    seg_t sg = {0}; if (mode & 2) { const size_t m = (size_t)1 << (K * ((NBb / K) < ((L + K - 1) / K) ? (NBb / K) : ((L + K - 1) / K)));
      sg.c0 = xmalloc(4 * m); sg.c1 = xmalloc(4 * m); sg.bp = xmalloc(m * (size_t)(NBb / K + (L + K - 1) / K + 2)); }
    uint32_t st[256], ss[256]; uint8_t sym[256];
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < ntiles; t++) {
      const float* xt = x + (size_t)t * 256;
      if (mode & 4) { beam_run(bm, xt, cb, 128, -1, st); const int c = (int)(st[128] >> K); beam_run(bm, xt, cb, 0, c, st); }
      else { vit_run(xt, cbs, Ls, K, 128, -1, vc, vn, vg, vbj, vbp, st);
             const int c = (int)(st[128] >> K); vit_run(xt, cbs, Ls, K, 0, c, vc, vn, vg, vbj, vbp, st); }
      for (int i = 0; i < 256; i++) sym[i] = (uint8_t)(st[i] & ((1u << K) - 1));
      states_from_syms(sym, L, K, ss);
      const float c0 = path_cost(xt, cb, ss); float m = c0;
      if (seedcost) seedcost[t] = c0;
      if ((mode & 5) == 1) { m = beam_run_seeded(bm, xt, cb, ss, st); for (int i = 0; i < 256; i++) sym[i] = (uint8_t)(st[i] & ((1u << K) - 1)); }
      if (mode & 2) m = refine(xt, cb, L, K, sym, NBb, &sg);
      states_from_syms(sym, L, K, st);
      for (int i = 0; i < 256; i++) out[(size_t)t * 256 + i] = cb[st[i]];
      if (cost) cost[t] = m;
    }
    free(vc); free(vn); free(vg); free(vbj); free(vbp); if (bm) beam_free(bm);
    if (sg.c0) { free(sg.c0); free(sg.c1); free(sg.bp); }
  }
}
