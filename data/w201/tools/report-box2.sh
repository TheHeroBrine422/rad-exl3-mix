#!/bin/bash
# report-box2.sh -- (w201) all the comparisons of the report, on box2 (CPU only): writes /data/q27/w201/eq/compare/*.json.
# Run with the venv python in isolated mode (-I). Each pair: REF (the coordinator's old score, or the earlier w201 run) vs NEW.
set -u
PY="/data/q27/venv/bin/python -I"; C=/data/q27/w201/tools/compare.py; G=/data/q27/w180/slices/golden-s195c2
OLD=/data/q27/srch183/scores; N=/data/q27/w201/eq; O=$N/compare
mkdir -p $O
$PY $C $G $OLD/p1-v3s183-2-manual-01645-s195c2-m     $N/new/p1-v3s183-2-manual-01645-s195c2-m     > $O/eq-2.0.json
$PY $C $G $OLD/p1s2-v3s183-4.5-manual-05175-s195c2-m $N/new/p1s2-v3s183-4.5-manual-05175-s195c2-m > $O/eq-4.5.json
$PY $C $G $OLD/p1s2-v3s183-7-inc-02366-s195c2-m      $N/new/p1s2-v3s183-7-inc-02366-s195c2-m      > $O/eq-7.0.json
$PY $C $G $N/new/p1-v3s183-2-manual-01645-s195c2-m   $N/prog/p1-v3s183-2-manual-01645-s195c2-m     > $O/progress-2.0.json
$PY $C $G $N/prefdfix/p1s2-v3s183-7-inc-02366-s195c2-m $N/new/p1s2-v3s183-7-inc-02366-s195c2-m    > $O/fdfix-neutral-7.0.json
$PY $C $G $N/new/p1-v3s183-2-manual-01645-s195c2-m   $N/smoke/smoke-2.0-madK1.625-s195c2-m         > $O/smoke-2.0-madK1.625.json
$PY /data/q27/w201/tools/progcheck.py $G $N/prog/p1-v3s183-2-manual-01645.progress $N/new/p1-v3s183-2-manual-01645-s195c2-m > $O/progcheck-2.0.json
echo REPORT DONE
