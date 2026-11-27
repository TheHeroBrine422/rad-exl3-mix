#!/bin/bash
# qenv.sh MACHINE  (sourced) -- the global QAD queue's per-machine paths. Same trainer / data / judge on both machines; only the paths differ.
#   QROOT queue root (jobs.txt, claims/, done/, runs/, scores/, q.log, results.tsv, code/)   IN  = w214's inputs (lv/, rel/, genmask, gold/, pools214.json, chunks214.txt)
#   CARDS  GPU indices this machine's queue may use    CARDLOCK(card) / RAMLOCK  flock files (lock order: card, then ram)    MINRAM  GB MemAvailable needed to start a training
MACHINE=${1:?machine (box2|iceman)}
case $MACHINE in
box2)
  Q=/data/q27; QROOT=$Q/queue; IN=$Q/w214; PY=$Q/venv/bin/python
  MODEL=$Q/models/qwen38-27b-bf16; CALIB=$Q/challenge/golden/calib
  CHUNKS=$IN/chunks214.txt; J217=$Q/w217/slices/golden-j217; CTLDIR=$Q/w217/scores; JLONG=$QROOT/slices/golden-jlong
  CARDS="0 1 2 3"; CARDLOCK() { echo /data/q27/locks/card$1.lock; }; RAMLOCK=/data/q27/locks/ram.lock; MINRAM=35
  RB=$Q/w206/install/bin/radiance; RH=$Q/w206/install/share/radiance; RLIBS=/opt/rocm-10.1/lib:$Q/ffmpeg-libs
  GPUENV() { echo "ROCR_VISIBLE_DEVICES=$1"; }
  FLA=$IN/pylib-fla;;
iceman)
  QROOT=/data/queue; IN=$QROOT/w214; PY=/data/venv-exl3/bin/python
  MODEL=/data/workspace/models/qwen38-27b-bf16; CALIB=/data/challenge/golden/calib
  CHUNKS=$IN/chunks214.iceman.txt; J217=$QROOT/slices/golden-j217; CTLDIR=$QROOT/ctl/scores; JLONG=$QROOT/slices/golden-jlong
  CARDS="0"; CARDLOCK() { echo /root/card$1.lock; }; RAMLOCK=/root/ram.lock; MINRAM=30
  RB=/data/w218/rad/install/bin/radiance; RH=/data/w218/rad/home; RLIBS=
  GPUENV() { echo "ROCR_VISIBLE_DEVICES=$1 HIP_VISIBLE_DEVICES=$1"; }
  FLA=$IN/pylib-fla;;
test)   # dispatcher self-test (no GPU): QROOT_OVERRIDE=/tmp/dir, CODE_PY = a fake trainer
  QROOT=${QROOT_OVERRIDE:?}; IN=$QROOT/in; PY=python3; MODEL=/none; CALIB=/none; CHUNKS=$IN/chunks; J217=$QROOT/j217; CTLDIR=$QROOT/ctl
  CARDS=${QTEST_CARDS:-"0 1"}; CARDLOCK() { echo $QROOT/card$1.lock; }; RAMLOCK=$QROOT/ram.lock; MINRAM=0; RB=/bin/false; RH=; RLIBS=; GPUENV() { echo "X=$1"; }; FLA=;;
*) echo "unknown machine $MACHINE" >&2; return 2 2>/dev/null || exit 2;;
esac
CODE=$QROOT/code; HERE=$CODE/v3/qadq
TRAINER=${CODE_PY:-$CODE/v3/qad220/qad220.py}
BUILD_LOCK=$QROOT/build.lock
export PYTHONPATH=$FLA QAD_PIPELINE=$CODE/pipeline TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=1        # w233: compile in the trainer process: the default pool is 32 worker processes per trainer (~16 GB host RAM each trainer)
qsay() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $QROOT/q.log; }
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:256   # w233 / w236 (11-27): the r6 finals were measured with this allocator setting
