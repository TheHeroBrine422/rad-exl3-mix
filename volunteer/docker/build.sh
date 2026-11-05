#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# build.sh [rocm|cuda] [COMMIT | --worktree] -- build the worker image from one committed rad-exl3-mix tree (git
# archive of only the paths the worker needs: never local/, data, untracked files or the dev shell scripts under
# v3/e3/; --worktree takes the same paths from the working tree, for tests only, and tags -dirty).
#
# Tags (IMAGE_REPO default rxm-worker, e.g. IMAGE_REPO=docker.io/<hub-user>/rxm-worker; VERSION from volunteer/VERSION):
#   $IMAGE_REPO:$VERSION-<stack>   e.g. 0.1.0-rocm7.14-gfx1201   immutable: what donors pin
#   $IMAGE_REPO:$VERSION-<variant> e.g. 0.1.0-rocm               moves with rebuilds of the same version (new stack)
#   $IMAGE_REPO:<variant>          e.g. rocm                     latest release of that variant
#   $IMAGE_REPO:<variant>-<commit>                               the exact source commit
# Publishing is a separate, manual step (README "Publishing the worker image"); this script never pushes.
set -euo pipefail
VAR=${1:-rocm}; R=$(cd "$(dirname "$0")/../.." && pwd)
DOCKER=${DOCKER:-docker}; REPO=${IMAGE_REPO:-rxm-worker}; VERSION=${VERSION:-$(cat "$R/volunteer/VERSION")}
case $VAR in                                   # keep in step with the ARGs of Dockerfile.$VAR
  rocm) STACK=rocm7.14-gfx1201 ;;
  cuda) STACK=cu128 ;;
  *) echo "variant: rocm | cuda"; exit 2 ;;
esac
PATHS="LICENSE NOTICE LICENSES quantize/ourq.py quantize/exl3-patches v3/e3/exl3frac v3/e3/exl3frac2 v3/e3/exl3cb volunteer"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
if [ "${2:-}" = "--worktree" ]; then
  C=$(git -C "$R" rev-parse --short HEAD)-dirty
  tar -C "$R" --exclude=__pycache__ --exclude='*.pyc' --exclude=.env --exclude=secrets --exclude='v3/e3/*/*.sh' \
      --exclude='v3/e3/*/build' -cf "$T/ctx.tar" $PATHS
else
  C=$(git -C "$R" rev-parse --short "${2:-HEAD}")
  git -C "$R" archive --format=tar "$C" -- $PATHS ':(exclude)v3/e3/**/*.sh' > "$T/ctx.tar"
fi
tar -C "$T" -xf "$T/ctx.tar" volunteer/docker/Dockerfile.$VAR 2>/dev/null || { echo "volunteer/ is not committed at $C"; exit 1; }
cp "$T/volunteer/docker/Dockerfile.$VAR" "$T/Dockerfile" && tar -C "$T" -rf "$T/ctx.tar" Dockerfile
TAG="$REPO:$VERSION-$STACK"
"$DOCKER" build --build-arg VERSION="$VERSION" --build-arg REPO_COMMIT="$C" --build-arg IMAGE_TAG="$TAG" \
  --build-arg CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "$TAG" -t "$REPO:$VERSION-$VAR" -t "$REPO:$VAR" -t "$REPO:$VAR-$C" ${DOCKER_BUILD_FLAGS:-} - < "$T/ctx.tar"
echo "built $TAG (also $REPO:$VERSION-$VAR, $REPO:$VAR, $REPO:$VAR-$C)"
