#!/bin/bash
# asym27 both arms + verdict (w78, 11-07): one entry point, resumable (rerun after any stop / restart).
# Per-arm exit status and the final "asym27: done" / "asym27: FAIL" go to /data/asym27/queue.log and /data/q1106/q.log.
export MINMEM_GB=4 TRITON_CACHE_DIR=/data/asym27/triton
exec /bin/bash /data/q1106/after-asym27.sh
