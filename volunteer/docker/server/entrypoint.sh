#!/bin/sh
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# rxmq -- the server image's entrypoint (on PATH, so `docker compose exec server rxmq admin list` works too).
#   rxmq [serve] [server.py flags]   the queue (config from RXMQ_* env; flags win)
#   rxmq admin COMMAND ...           admin.py on the same db / data / results (RXMQ_DB, RXMQ_DATA, RXMQ_RESULTS)
#   rxmq healthcheck                 exit 0 if /healthz answers
#   rxmq version                     image version and repo commit
set -eu
V=/opt/rxmq/volunteer
cmd=${1:-serve}
[ $# -gt 0 ] && shift
case "$cmd" in
  serve)       exec python -I "$V/server/server.py" "$@" ;;
  admin)       exec python -I "$V/server/admin.py" "$@" ;;
  healthcheck) exec python -I /opt/rxmq/healthcheck.py ;;
  version)     cat /opt/rxmq/IMAGE ;;
  -*)          exec python -I "$V/server/server.py" "$cmd" "$@" ;;
  *)           echo "usage: rxmq [serve [flags] | admin COMMAND ... | healthcheck | version]" >&2; exit 2 ;;
esac
