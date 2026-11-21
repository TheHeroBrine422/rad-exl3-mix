#!/bin/bash
# pull.sh -- copy w210's raw outputs from box2:/data/q27/w210/out into raw/. W210_RSH = the ssh command for box2
# and W210_HOST = user@host (rad-local/workers/w210/pull.sh sets both; the address and key stay out of the repo).
cd "$(dirname "$0")"
rsync -a --exclude '*.u32' --exclude '*.f16' --exclude '*.f32' --exclude '*.safetensors' --exclude 'err' --exclude '*.err' --exclude '*.part' --exclude 'done' \
  -e "${W210_RSH:?set W210_RSH to the box2 ssh command}" "${W210_HOST:?set W210_HOST to user@host of box2}":/data/q27/w210/out/ raw/ < /dev/null
