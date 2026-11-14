# search/refit/w158 — #173-03: A8 calibration rows on the mad ladder, M1m refit, re-solve, interaction test (w158, 2026-11-14)

Runbook step 3 / 3b (`rad-local/RUNBOOK-overnight-1113.md`), redo of w143's #173-02 on the final format (mad trunk, trellis
embedding in the budget, a/b T4 / T5, A8). Rules: `RULES-173-03.md` (a91e446) and `RULES-173-03b.md` (ac05ee7), each committed
alone before its data. Results: LOG #173-03, data `data/a8rows173/`.

| file | what |
|---|---|
| `mkmenu173.py` | the menu (ds8): trunk = mad level dirs + packs-mad (bytes = the pack's entries, err = proxy × diag R), head = hcurve151's mad + mul1 plans (i8 not offered), embedding = w157's block, fixed bytes from a real mix (vision tower dropped), a/b T5 extra |
| `hdonor173.sh` | served exl3i R16 head donors `hd-<plan>.rad` (w151's pack path + slim radmix, census `--expect-head`) |
| `run173.py` | ds8 coordinator: mixfile (trunk packs, head / embedding donors, a/b drop + ssm_ab add, vision drop, A8 act lines) → radmix (conv.lock) → census (patched: mad = v4) → score on the first free card (ds8 HIP N, box2 HIP N over the 10 GbE link; `guard` cards only while no w151 process runs) → container deleted. Restartable |
| `mkspecs173.py` | RULES-173-03's rows (wave 1: U 20, G 48, H 7, E 7, B 2, F 4; wave 2: S1 5, S2 12) |
| `docsums158.py`, `collect158.py` | per-document KL sums (ds8) → fit obs (gen-masked KLD, delta-method SE vs U-2.5, swap rows' base and paired SE) |
| `m158.py` | M1m = m133's XModel + `embf` (embedding priced on the menu err, one parameter), `v2x` (V2-era prior sds × F), `lin2` (sub-1 F3), A8 as the reference mode (f16 twins = "x16", signed c) |
| `fin158.py` | fit / CV / pred / curve / solve (mckp on the menu, starts = uniform below the size + the previous size's pick, a/b T5 bytes at ≥ 5.5) |
| `fits173.sh`, `models173.json` | colo runner (wave-1 fits + CVs; the sub-1 rule F1 / F2 / F3) |
| `pickcmp158.py` | pick movement between two fits' picks |
| `draw173b.py`, `mkinter173.py`, `inter173.py` | the interaction test: fresh slice draw, 77 specs from the 2.5 / 4.0 picks, the I / ρ table |
| `census-mad-v4.patch` | for the lead: `store/census.py` reports mad weights as encoding v1 / v3; mad is v4 (exl3i_encoding.h) |

Run (ds8 `/data/q27/w158/`, colo `/data/work/w158/`):
```
python3 mkmenu173.py menus/menu173m-v1.json --levels <ladder-mad K>=1 dirs> --packs /data/q27/packs-mad --diag <k2.5 level> \
   --heads /data/q27/mad/heads-rep/hcurve151.json --head-dir /data/q27/w158/heads --emb-menu menu-emb17403.json \
   --fixed-from <a census-checked mix> --ab-t4 /data/q27/mad/ab/donor-ab-K4.rad --ab-t5 /data/q27/mad/ab/donor-ab-K5.rad
python3 mkspecs173.py MENU specs-w1.jsonl 1        # wave 2: ... specs-w2.jsonl 2 (menu with the sub-1 levels)
setsid nohup python -I run173.py conf-w1.json &    # cards.txt lines "ds8:7", "box2:1", "ds8:0 guard", ...
python -I docsums158.py run/w1 fit129 GOLDEN docsums-w1.json; python3 collect158.py MENU obs-w1.jsonl specs-w1.jsonl --docs docsums-w1.json
bash fits173.sh w1 ../obs-w1.jsonl                 # colo; then fin158.py solve ... --min-k 1 --save-dir picks-M1m-in
```
