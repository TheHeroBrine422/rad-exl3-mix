#!/usr/bin/env python3
"""rreport.py -- score RULER runs (stdlib only).

    python3 -I rreport.py report RUNDIR [--select SEL.json | --cells SPEC] [--json OUT]
        per task x length score (RULER's x100 scale), per-length averages, the overall score (mean over lengths of
        the mean over tasks), a stratified bootstrap 95% interval on the overall
    python3 -I rreport.py compare RUN_A RUN_B [--select ... | --cells ...] [--json OUT]
        B - A paired on the samples both runs hold: overall / per length / per task / per cell differences with a
        paired bootstrap 95% interval, and flip counts
    python3 -I rreport.py same RUN_A RUN_B      determinism: are the replies byte-identical on the common samples?
Exit status of `same`: 0 identical, 1 not.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib


def restrict(results, a):
    if getattr(a, 'select', None):
        cells = rlib.load_selection(a.select)['cells']
    elif getattr(a, 'cells', None):
        cells = rlib.parse_cells(a.cells)
    else:
        return results
    keep = {(c, i) for c, ix in cells.items() for i in ix}
    return [r for r in results if (r['cell'], r['index']) in keep]


def meta_of(rundir):
    rj = json.load(open(os.path.join(rundir, 'run.json')))
    return {'run': os.path.abspath(rundir), 'name': rj.get('name'), 'identity': rj['identity']}


def table(rep):
    lengths = sorted({c['length'] for c in rep['cells'].values()})
    tasks = sorted({c['task'] for c in rep['cells'].values()}, key=rlib.task_rank)
    w = max(len(t) for t in tasks + ['overall'])
    lines = [f"{'task':<{w}} " + ' '.join(f'{L // 1024:>6}k' for L in lengths)]
    for t in tasks:
        row = []
        for L in lengths:
            c = rep['cells'].get(f'{t}_{L}')
            row.append(f"{c['score']:6.1f}{'*' if c['n'] < 10 else ' '}" if c else '      -')
        lines.append(f'{t:<{w}} ' + ' '.join(row))
    lines.append(f"{'avg':<{w}} " + ' '.join(f"{rep['by_length'][str(L)]:6.1f} " for L in lengths))
    ci = rep.get('overall_ci95')
    lines.append(f"overall {rep['overall']:.2f}" + (f"  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else '') +
                 f"  ({rep['n']} samples; * = cell with < 10 samples)" +
                 ('' if rep.get('complete_grid', True) else '\nNOTE: the lengths hold different task sets: averages '
                  'are not comparable across lengths'))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('report', 'compare', 'same'):
        p = sub.add_parser(name)
        p.add_argument('runs', nargs=1 if name == 'report' else 2)
        p.add_argument('--select')
        p.add_argument('--cells')
        p.add_argument('--json')
        p.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == 'report':
        res = restrict(rlib.run_results(a.runs[0]), a)
        rep = rlib.report(res, meta=meta_of(a.runs[0]), boot=a.boot)
        print(table(rep))
        out = rep
    elif a.cmd == 'compare':
        ra, rb = (restrict(rlib.run_results(r), a) for r in a.runs)
        ia, ib = (json.load(open(os.path.join(r, 'run.json')))['identity'] for r in a.runs)
        if ia['set'] != ib['set'] or ia['template'] != ib['template']:
            sys.exit('the runs used different sets or templates: not paired')
        sa = {r['key']: r.get('cell_sha256') for r in ra}
        bad = [r['key'] for r in rb if r['key'] in sa and sa[r['key']] != r.get('cell_sha256')]
        if bad:
            sys.exit(f'{len(bad)} common samples come from different rows ({bad[:3]}): not paired')
        out = rlib.paired(ra, rb, boot=a.boot, meta={'a': meta_of(a.runs[0]), 'b': meta_of(a.runs[1])})
        print(f"A {out['a_overall']:.2f}  B {out['b_overall']:.2f}  B-A {out['diff']:+.2f}  95% CI "
              f"[{out['diff_ci95'][0]:+.2f}, {out['diff_ci95'][1]:+.2f}]  ({out['n_paired']} paired samples; flips "
              f"A>B {out['flips_a_right_b_wrong']}, B>A {out['flips_b_right_a_wrong']})")
        for L, v in out['by_length'].items():
            print(f"  {int(L) // 1024:>4}k  A {v['a']:6.2f}  B {v['b']:6.2f}  {v['diff']:+6.2f}  "
                  f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
        for t, v in out['by_task'].items():
            print(f"  {t:<16} A {v['a']:6.2f}  B {v['b']:6.2f}  {v['diff']:+6.2f}  "
                  f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    else:
        ra, rb = ({r['key']: r for r in restrict(rlib.run_results(x), a)} for x in a.runs)
        common = sorted(set(ra) & set(rb))
        diff = [k for k in common if ra[k]['pred'] != rb[k]['pred']]
        out = {'common': len(common), 'different': len(diff), 'keys': diff[:50]}
        print(f'{len(common)} common samples, {len(diff)} with different replies' +
              (': ' + ', '.join(diff[:10]) if diff else ''))
        if a.json:
            rlib.write_json(a.json, out)
        sys.exit(1 if diff or not common else 0)
    if a.json:
        rlib.write_json(a.json, out)


if __name__ == '__main__':
    main()
