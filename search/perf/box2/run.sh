#!/bin/bash
# b2run.sh CARD TAG MIX GOLDEN [radiance args] -- one locked timing run in /data/q27/w123 (box2 / ds8)
cd /data/q27/w123; C=$1 T=$2 M=$3 G=$4; shift 4
exec flock -n /data/q27/locks/card$C.lock flock -s /data/q27/locks/ram.lock ./tscore.sh /data/q27/rad/install /data/q27/rad/home $M $G scores/$T $C "$@"
