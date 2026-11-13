#!/usr/bin/env python3
"""lintsum.py DIR [DIR ...] -- radisa check JSON reports (lint.sh) summed: per report status + error / warning counts,
then every error / warning grouped by (rule, source file:line) with the kernels and instruction it hits. (w137, #175)"""
import collections, glob, json, os, sys

def kname(n):
    return n if len(n) < 70 else n[:67] + '...'

for d in sys.argv[1:]:
    print(f'## {d}')
    groups = collections.defaultdict(lambda: {'n': 0, 'kern': set(), 'ins': collections.Counter(), 'msg': '', 'units': set()})
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        r = json.load(open(f)); u = os.path.basename(f)[:-5]
        c = collections.Counter(x['severity'] for k in r.get('kernels', []) for x in k.get('findings', []))
        c.update(x['severity'] for x in r.get('findings', []))
        print(f"{u:60s} {r.get('status','?'):12s} kernels {len(r.get('kernels', [])):4d}  E {c['error']:4d}  W {c['warning']+c['review']:3d}")
        for k in r.get('kernels', []):
            for x in k.get('findings', []):
                if x['severity'] not in ('error', 'warning', 'review'):
                    continue
                loc = x.get('loc') or {}
                key = (x['severity'], x['rule'], f"{os.path.basename(loc.get('file','?'))}:{loc.get('line','?')}")
                g = groups[key]; g['n'] += 1; g['kern'].add(k['name']); g['units'].add(u)
                g['ins'][x.get('instruction', '').split(' ')[0]] += 1
                g['msg'] = g['msg'] or x['message']
    print()
    for (sev, rule, loc), g in sorted(groups.items(), key=lambda t: (t[0][0], t[0][1], -t[1]['n'])):
        print(f"{sev:7s} {rule:26s} {loc:34s} n {g['n']:4d} kernels {len(g['kern']):3d} units {len(g['units'])}  ins {dict(g['ins'].most_common(3))}")
        print(f"        e.g. {kname(sorted(g['kern'])[0])}: {g['msg'][:220]}")
    print()
