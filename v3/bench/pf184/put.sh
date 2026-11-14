#!/bin/bash
# put.sh TAG [TREE] -- the w160 radiance worktree (tracked files, uncommitted edits included: git stash create) as box2
# /data/q27/w160/src/TAG (COMMIT file = the tree's HEAD + "dirty" when edited). (#184, w160)
set -e; T=$1; W=${2:-/workspace/wt/w160-radiance}; B=/workspace/rad-local/bin/ssh-box2.sh
P=/workspace/rad-local/workers/w117/box2-put.sh
git -C $W add -A plugin >/dev/null; REV=$(git -C $W stash create); [ -n "$REV" ] || REV=HEAD
D=$(git -C $W rev-parse --short HEAD)$( [ "$REV" = HEAD ] || echo -dirty)
$B "mkdir -p /data/q27/w160/src"
git -C $W archive --format=tar.gz $REV | $P /data/q27/w160/src/$T.tgz
$B "cd /data/q27/w160/src && rm -rf $T && mkdir $T && tar xzf $T.tgz -C $T && echo '$D' > $T/COMMIT && rm $T.tgz && echo put $T $D"
