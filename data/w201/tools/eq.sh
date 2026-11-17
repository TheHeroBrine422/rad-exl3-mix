#!/bin/bash
# eq.sh -- (w201) the equivalence queue on box2 card 3: the three s195c2 mixes (2.0, 4.5, 7.0) scored with the w201 build by
# the coordinator's score command (host.json score_cmd), then one progress run (score_cmd_progress, --kld-progress) of the
# 2.0 mix. Restartable: score-w201.sh skips a finished output. Outputs under /data/q27/w201/eq/{new,prog}/.
set -u
NEW=/data/q27/w201/rad21
G=/data/q27/w180/slices/golden-s195c2
M=/data/q27/srch183/mixes
E=/data/q27/w201/eq
S=/data/q27/w201/tools/score-w201.sh
bash $S $NEW $M/p1-v3s183-2-manual-01645.mix       $G $E/new/p1-v3s183-2-manual-01645-s195c2-m
bash $S $NEW $M/p1s2-v3s183-4.5-manual-05175.mix   $G $E/new/p1s2-v3s183-4.5-manual-05175-s195c2-m
bash $S $NEW $M/p1s2-v3s183-7-inc-02366.mix        $G $E/new/p1s2-v3s183-7-inc-02366-s195c2-m
bash $S $NEW $M/p1-v3s183-2-manual-01645.mix       $G $E/prog/p1-v3s183-2-manual-01645-s195c2-m $E/prog/p1-v3s183-2-manual-01645.progress
echo "EQ QUEUE DONE"
