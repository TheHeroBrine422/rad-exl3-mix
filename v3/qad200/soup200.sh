#!/bin/bash
# soup200.sh NAME CARD MEMBER... (w187, #200, box2) -- the soup of trained log multipliers (soup198.py build mean; exp on
# card CARD's GPU under its lock), then its served read (eval196.py, card CARD). Restartable (a finished soup is skipped).
NAME=$1 CARD=$2; shift 2
cd /data/q27/w187 && source env.sh
SOUP=$CODE/v3/qad200/soup198.py
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $W/q200.log; }
if [ ! -e soups/$NAME/.done ]; then
  flock /data/q27/locks/card$CARD.lock env ROCR_VISIBLE_DEVICES=$CARD $PY $SOUP build soups/$NAME mean "$@" --start $START --device cuda:0 \
    > logs/soup-$NAME.log 2>&1 || { say "soup $NAME FAILED (logs/soup-$NAME.log)"; exit 1; }
  say "soup $NAME built: mean of $# members (exp on card $CARD)"
fi
if [ ! -e served/$NAME/result.json ]; then
  $PY $CODE/v3/qad200/eval196.py $NAME served --vlevel $VL --template $TPL --host $HOSTC --pack-home /data/q27/rad10-e157b/home \
    --card $CARD --ckpt $W/soups/$NAME --slice x181g=$G1 --slice fe38=$G2 > logs/served-$NAME.out 2>&1
  rc=$?; say "served $NAME rc $rc"; [ $rc -eq 0 ] || exit 1
fi
