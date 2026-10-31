#!/bin/bash
# build.sh -- hproxy, headdec and radmix against radiance's headers and the plugin's EXL3 host decoder.
# RXM_PLUGIN: the exl3import plugin directory of the qc-exl3-ladder plugins (radiance PR #292).
: ${RXM_WORK:=/root} ${RXM_RADIANCE:=$RXM_WORK/install-card} ${RXM_PLUGIN:=$RXM_WORK/radiance-hero/plugin/exl3import}; export RXM_WORK RXM_RADIANCE RXM_PLUGIN   # paths (README: Paths)
set -e
cd "$(dirname "$0")"
P=${RXM_PLUGIN}
INC=$(dirname $(find ${RXM_RADIANCE}/include -name rad_format.h | head -1))
gcc -O3 -march=native -fopenmp -c $P/common/exl3_decode.c -I$P/common -I$P/include -I$INC -o exl3_decode.o
hipcc -O3 -march=native -fopenmp -std=c++17 -c hproxy.cpp -I$P/common -I$P/include -I$INC -o hproxy.o
hipcc -fopenmp hproxy.o exl3_decode.o -lhipblas -o hproxy
g++ -O3 -march=native -fopenmp -std=c++17 headdec.cpp exl3_decode.o -I$P/common -I$P/include -I$INC -o headdec
g++ -O2 -std=c++17 radmix.cpp -I$INC -o radmix
echo built
