#!/bin/bash
# pull.sh -- copy w210's raw outputs from box2:/data/q27/w210/out into raw/. W210_RSH = the ssh command for box2
# (rad-local/workers/w210/pull.sh sets it; the address and key stay out of the repo).
cd "$(dirname "$0")"
rsync -a --exclude '*.u32' --exclude '*.f16' --exclude '*.f32' --exclude '*.safetensors' --exclude 'err' --exclude '*.err' --exclude '*.part' --exclude 'done' \
  -e "${W210_RSH:?set W210_RSH to the box2 ssh command}" root@127.0.0.1:/data/q27/w210/out/ raw/ < /dev/null
