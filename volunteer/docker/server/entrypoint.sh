#!/bin/sh
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# radquant -- the server image's entrypoint (on PATH, so `docker compose exec server radquant admin list` works too;
# `rxmq` is the v0 name of the same script, kept for 0.2).
#   radquant [serve] [server.py flags]   the queue (config from RADQUANT_* env; flags win)
#   radquant admin COMMAND ...           admin.py on the same db / data / results (RADQUANT_DB, _DATA, _RESULTS)
#   radquant healthcheck                 exit 0 if /healthz answers
#   radquant version                     image version and repo commit
set -eu
V=/opt/radquant/volunteer
cmd=${1:-serve}
[ $# -gt 0 ] && shift
case "$cmd" in
  serve)       exec python -I "$V/server/server.py" "$@" ;;
  admin)       exec python -I "$V/server/admin.py" "$@" ;;
  healthcheck) exec python -I /opt/radquant/healthcheck.py ;;
  version)     cat /opt/radquant/IMAGE ;;
  -*)          exec python -I "$V/server/server.py" "$cmd" "$@" ;;
  *)           echo "usage: radquant [serve [flags] | admin COMMAND ... | healthcheck | version]" >&2; exit 2 ;;
esac
