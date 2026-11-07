"""rlib -- the RULER harness's shared pieces (stdlib only; numpy not needed): sample sets and selections, prompt
rendering, RULER's metrics, per-sample result files, reports and paired comparisons.

Imported by rrun.py and rreport.py, and meant to be imported by other harnesses (kv_judge's T2, the LSA gate):
    import rlib
    samples = rlib.select(setdir, rlib.load_selection('select/fast-v1.json'))   # list of row dicts + 'cell' / 'key'
    prompt  = rlib.render(row, 'chat')                                           # text with Qwen's chat markup
    s       = rlib.score(row, reply_text)                                        # RULER's string match, 0..1
    rep     = rlib.report(results)            # results: [{'task','length','index','score'}, ...]
    cmp     = rlib.paired(results_a, results_b)
"""
from __future__ import annotations

import gzip, hashlib, json, math, os, random, re
from typing import Any, Callable

Json = dict[str, Any]

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_SCHEMA = 'ruler-report/1'
COMPARE_SCHEMA = 'ruler-compare/1'

# which RULER metric a task uses (scripts/eval/synthetic/constants.py): qa = any reference counts, else the fraction
PART_TASKS = ('qa_squad', 'qa_hotpot')
TASK_ORDER = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2',
              'niah_multikey_3', 'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_squad', 'qa_hotpot']

# Qwen3.5 / 3.8 chat markup for one user turn with thinking off (both models' templates render exactly this with
# enable_thinking=false, add_generation_prompt=true); the answer prefix follows as the assistant's opening words.
QWEN_USER = '<|im_start|>user\n'
QWEN_ASSIST = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


# ---------------------------------------------------------------- io
def write_atomic(path: str, data: str | bytes) -> None:
    """w41's convention: write PATH.part, fsync, rename; a reader trusts only the final name."""
    if isinstance(data, str):
        data = data.encode()
    tmp = f'{path}.part'
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: str, obj: Any) -> None:
    write_atomic(path, json.dumps(obj, indent=1, sort_keys=True) + '\n')


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- sets and selections
def manifest(setdir: str) -> Json:
    with open(os.path.join(setdir, 'manifest.json')) as f:
        out: Json = json.load(f)
    return out


def manifest_id(setdir: str) -> Json:
    """The set's identity (stable while cells are added to it): name, seed, format, upstream generator."""
    m = manifest(setdir)
    return {'name': m['name'], 'seed': m['seed'], 'format': m['format'], 'upstream': m['upstream']}


def cell_sha256(setdir: str, cell: str) -> str:
    """The content hash of one cell's uncompressed rows (the manifest's)."""
    return str(manifest(setdir)['cells'][cell]['sha256'])


def cells_sha256(setdir: str, cells: list[str] | dict[str, Any]) -> str:
    """One hash over the named cells' content hashes: what a selection's rows are (pin it in a judge's profile)."""
    m = manifest(setdir)['cells']
    return sha256_bytes(json.dumps({c: m[c]['sha256'] for c in sorted(cells)}, sort_keys=True).encode())


_cells: dict[tuple[str, str], list[Json]] = {}


def cell_rows(setdir: str, cell: str) -> list[Json]:
    key = (os.path.abspath(setdir), cell)
    if key not in _cells:
        p = os.path.join(setdir, cell + '.jsonl.gz')
        raw = gzip.open(p, 'rb').read()
        want = manifest(setdir)['cells'][cell]['sha256']
        if sha256_bytes(raw) != want:
            raise ValueError(f'{p}: content hash differs from the manifest')
        _cells[key] = [json.loads(l) for l in raw.splitlines()]
    return _cells[key]


def parse_cells(spec: str) -> dict[str, list[int]]:
    """'task:L:a-b,task:L:n,...' (a-b = row range, n = first n rows; task or L may be '*' with --set) -> dict."""
    out: dict[str, list[int]] = {}
    for part in spec.split(','):
        if not part:
            continue
        t, L, r = part.split(':')
        a, b = (int(x) for x in r.split('-')) if '-' in r else (0, int(r))
        out.setdefault(f'{t}_{L}', []).extend(range(a, b))
    return {k: sorted(set(v)) for k, v in out.items()}


def load_selection(path: str) -> Json:
    """A selection file: {"name", "set": set name, "cells": {"task_L": [indices] | N (first N) | "a-b"}}."""
    with open(path) as f:
        s: Json = json.load(f)
    cells: dict[str, list[int]] = {}
    for k, v in s['cells'].items():
        if isinstance(v, int):
            cells[k] = list(range(v))
        elif isinstance(v, str):
            a, b = (int(x) for x in v.split('-'))
            cells[k] = list(range(a, b))
        else:
            cells[k] = sorted(v)
    s['cells'] = cells
    return s


def split_cell(cell: str) -> tuple[str, int]:
    t, L = cell.rsplit('_', 1)
    return t, int(L)


def select(setdir: str, cells: dict[str, Any]) -> list[Json]:
    """Rows of the selected cells ({cell: [rows] or N = the first N}): each row dict gains 'task', 'cell', 'key'
    (cell/row) and 'cell_sha256'."""
    out: list[Json] = []
    for cell in sorted(cells, key=lambda c: (split_cell(c)[1], task_rank(split_cell(c)[0]))):
        rows = cell_rows(setdir, cell)
        t, L = split_cell(cell)
        v = cells[cell]
        for i in (range(v) if isinstance(v, int) else v):
            if i >= len(rows):
                raise ValueError(f'{cell}: row {i} not in the set ({len(rows)} rows)')
            r = dict(rows[i])
            r.update(task=t, cell=cell, key=f'{cell}/{i:04d}', cell_sha256=cell_sha256(setdir, cell))
            out.append(r)
    return out


def task_rank(t: str) -> int:
    return TASK_ORDER.index(t) if t in TASK_ORDER else len(TASK_ORDER)


# ---------------------------------------------------------------- prompts
def render(row: Json, template: str) -> str:
    """base: RULER's own prompt (input + answer prefix, as for a base model);
    chat: one Qwen user turn (thinking off) holding the input, the assistant reply opened by the answer prefix
    without its surrounding spaces (as the LSA repo serves Qwen's RULER)."""
    if template == 'base':
        return str(row['input']) + str(row['answer_prefix'])
    if template == 'chat':
        return QWEN_USER + str(row['input']) + QWEN_ASSIST + str(row['answer_prefix']).strip()
    raise ValueError(template)


# ---------------------------------------------------------------- RULER's metrics (scripts/eval)
_np = re.compile(r'[\x00-\x1f]')


def postprocess_pred(s: str) -> str:
    """scripts/eval/evaluate.py: strip, every control character to a newline, strip."""
    return _np.sub('\n', s.strip()).strip()


def score(row: Json, reply: str) -> float:
    """One sample's RULER score in [0, 1]: string_match_part (qa) or string_match_all (the rest)."""
    pred = postprocess_pred(reply).lower()
    refs: list[str] = list(row['outputs'])
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    return max(hits) if row['task'] in PART_TASKS else sum(hits) / len(refs)


# ---------------------------------------------------------------- reports
def _mean(x: list[float]) -> float:
    return sum(x) / len(x) if x else float('nan')


def _group(results: list[Json]) -> dict[tuple[str, int], dict[int, float]]:
    cells: dict[tuple[str, int], dict[int, float]] = {}
    for r in results:
        cells.setdefault((r['task'], int(r['length'])), {})[int(r['index'])] = float(r['score'])
    return cells


def _weighted(cells_vals: dict[tuple[str, int], float]) -> tuple[float, dict[int, float], dict[str, float]]:
    """RULER's aggregation: a cell's score is its mean; a length's the mean of its tasks' cells; overall the mean over
    lengths. cells_vals: {(task, L): mean}. Returns (overall, by_length, by_task)."""
    by_len: dict[int, list[float]] = {}
    by_task: dict[str, list[float]] = {}
    for (t, L), v in cells_vals.items():
        by_len.setdefault(L, []).append(v)
        by_task.setdefault(t, []).append(v)
    bl = {L: _mean(v) for L, v in sorted(by_len.items())}
    bt = {t: _mean(v) for t, v in sorted(by_task.items(), key=lambda x: task_rank(x[0]))}
    return _mean(list(bl.values())), bl, bt


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def report(results: list[Json], meta: Json | None = None, boot: int = 2000, seed: int = 0) -> Json:
    """Per task x length score (x100, RULER's scale), per-length averages, the overall score, with a sample bootstrap
    (stratified by cell) for the overall's 95% interval; cells carry a Wilson interval on the all-correct rate."""
    cells = _group(results)
    cv = {c: _mean(list(v.values())) for c, v in cells.items()}
    overall, bl, bt = _weighted(cv)
    grid: dict[int, set[str]] = {}
    for (t, L) in cells:
        grid.setdefault(L, set()).add(t)
    out: Json = {'schema': REPORT_SCHEMA, 'n': sum(len(v) for v in cells.values()), 'overall': 100 * overall,
           'complete_grid': len({frozenset(v) for v in grid.values()}) <= 1,
           'by_length': {str(L): 100 * v for L, v in bl.items()}, 'by_task': {t: 100 * v for t, v in bt.items()},
           'cells': {}}
    for (t, L), v in sorted(cells.items(), key=lambda x: (x[0][1], task_rank(x[0][0]))):
        vals = list(v.values())
        full = sum(1 for x in vals if x >= 1.0)
        out['cells'][f'{t}_{L}'] = {'task': t, 'length': L, 'n': len(vals), 'score': 100 * _mean(vals),
                                    'all_correct': full, 'wilson95_all_correct': list(wilson(full, len(vals)))}
    if boot and cells:
        rng = random.Random(seed)
        keys = list(cells)
        arr = {c: list(cells[c].values()) for c in keys}
        bs: list[float] = []
        for _ in range(boot):
            bv: dict[tuple[str, int], float] = {}
            for c in keys:
                a = arr[c]
                bv[c] = sum(a[rng.randrange(len(a))] for _ in a) / len(a)
            bs.append(100 * _weighted(bv)[0])
        bs.sort()
        out['overall_ci95'] = [bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1]]
        out['overall_se'] = _sd(bs)
    if meta:
        out['meta'] = meta
    return out


def _sd(x: list[float]) -> float:
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / max(1, len(x) - 1))


def paired(results_a: list[Json], results_b: list[Json], boot: int = 2000, seed: int = 0,
           meta: Json | None = None) -> Json:
    """B - A on the samples both hold (same task, length, row): the RULER-weighted difference of the overall score,
    per length, per task and per cell, each with a paired bootstrap 95% interval (resampling rows within each cell,
    the same rows for A and B), plus flip counts (rows A gets fully right and B not, and the reverse)."""
    ca, cb = _group(results_a), _group(results_b)
    common = {c: sorted(set(ca[c]) & set(cb[c])) for c in ca if c in cb}
    common = {c: v for c, v in common.items() if v}
    if not common:
        raise ValueError('no samples in common')
    A = {c: [ca[c][i] for i in idx] for c, idx in common.items()}
    B = {c: [cb[c][i] for i in idx] for c, idx in common.items()}

    def stats(sel: dict[tuple[str, int], list[int]]) -> tuple[float, float, dict[int, float], dict[int, float],
                                                              dict[str, float], dict[str, float]]:
        va = {c: _mean(A[c]) for c in sel}
        vb = {c: _mean(B[c]) for c in sel}
        oa, la, ta = _weighted(va)
        ob, lb, tb = _weighted(vb)
        return oa, ob, la, lb, ta, tb

    oa, ob, la, lb, ta, tb = stats(common)
    rng = random.Random(seed)
    keys = sorted(common)
    bo: list[float] = []
    bl: dict[int, list[float]] = {L: [] for L in la}
    bt: dict[str, list[float]] = {t: [] for t in ta}
    bc: dict[tuple[str, int], list[float]] = {c: [] for c in keys}
    for _ in range(boot):
        va: dict[tuple[str, int], float] = {}
        vb: dict[tuple[str, int], float] = {}
        for c in keys:
            n = len(A[c])
            ix = [rng.randrange(n) for _ in range(n)]
            va[c] = sum(A[c][i] for i in ix) / n
            vb[c] = sum(B[c][i] for i in ix) / n
            bc[c].append(100 * (vb[c] - va[c]))
        xa, ya, za = _weighted(va)
        xb, yb, zb = _weighted(vb)
        bo.append(100 * (xb - xa))
        for L in la:
            bl[L].append(100 * (yb[L] - ya[L]))
        for t in ta:
            bt[t].append(100 * (zb[t] - za[t]))

    def ci(v: list[float]) -> list[float]:
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    flips_ab = sum(1 for c in keys for a, b in zip(A[c], B[c]) if a >= 1.0 and b < 1.0)
    flips_ba = sum(1 for c in keys for a, b in zip(A[c], B[c]) if b >= 1.0 and a < 1.0)
    out: Json = {'schema': COMPARE_SCHEMA, 'n_paired': sum(len(v) for v in common.values()),
           'a_overall': 100 * oa, 'b_overall': 100 * ob, 'diff': 100 * (ob - oa), 'diff_ci95': ci(bo),
           'diff_se': _sd(bo), 'flips_a_right_b_wrong': flips_ab, 'flips_b_right_a_wrong': flips_ba,
           'identical_scores': all(a == b for c in keys for a, b in zip(A[c], B[c])),
           'by_length': {str(L): {'a': 100 * la[L], 'b': 100 * lb[L], 'diff': 100 * (lb[L] - la[L]), 'ci95': ci(bl[L])}
                         for L in la},
           'by_task': {t: {'a': 100 * ta[t], 'b': 100 * tb[t], 'diff': 100 * (tb[t] - ta[t]), 'ci95': ci(bt[t])}
                       for t in ta},
           'cells': {f'{t}_{L}': {'n': len(A[(t, L)]), 'a': 100 * _mean(A[(t, L)]), 'b': 100 * _mean(B[(t, L)]),
                                  'diff': 100 * (_mean(B[(t, L)]) - _mean(A[(t, L)])), 'ci95': ci(bc[(t, L)])}
                     for (t, L) in sorted(keys, key=lambda x: (x[1], task_rank(x[0])))}}
    if meta:
        out['meta'] = meta
    return out


# ---------------------------------------------------------------- run directories
def run_results(rundir: str) -> list[Json]:
    """Every finished sample of a run directory (pred/<cell>/<index>.json)."""
    out: list[Json] = []
    pd = os.path.join(rundir, 'pred')
    if not os.path.isdir(pd):
        return out
    for cell in sorted(os.listdir(pd)):
        for f in sorted(os.listdir(os.path.join(pd, cell))):
            if f.endswith('.json'):
                with open(os.path.join(pd, cell, f)) as fh:
                    out.append(json.load(fh))
    return out
