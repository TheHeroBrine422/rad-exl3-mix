#!/bin/bash
# rocm10-get.sh [DEST] -- ROCm 10.1.0 beside the pip SDK, from the image radiance's Dockerfile builds in
# (rocm/dev-ubuntu-24.04:10.1.0-full). The image's /opt/rocm is a directory of /etc/alternatives links around one
# self-contained root, /opt/rocm/core-10.1 (bin, lib, lib/llvm, include, share; 21 GB): that root is copied to DEST
# (default /opt/rocm-10.1) and is the ROCM_PATH.
# The torch venv and the pip SDK are not touched; no /opt/rocm is created (torch's cpp_extension would find it).
# The image is removed afterwards unless KEEP=1 (it sits on /, ~2x the tree). (w130, #170)
set -e; D=${1:-/opt/rocm-10.1}; IMG=rocm/dev-ubuntu-24.04:10.1.0-full
[ -f $D/.w130-ok ] && { echo "$D present"; exit 0; }
docker pull $IMG
c=$(docker create $IMG true); trap 'docker rm -f $c > /dev/null' EXIT
r=$(docker run --rm $IMG readlink -f /opt/rocm/core-10)      # /opt/rocm/core-10.1
rm -rf $D.part; mkdir -p $D.part
docker cp $c:$r/. $D.part/
docker run --rm $IMG sh -c '/opt/rocm/lib/llvm/bin/clang --version | head -2; ls /opt/rocm/core-10.1/.info; cat /opt/rocm/core-10.1/.info/version* 2>/dev/null; true' > $D.part/.w130-src
echo "$IMG $(docker image inspect -f '{{json .RepoDigests}}' $IMG 2>&1 || true) $r" >> $D.part/.w130-src
rm -rf $D; mv $D.part $D; touch $D/.w130-ok
[ "${KEEP:-0}" = 1 ] || docker rmi $IMG
echo "ROCM10 READY $D"; cat $D/.w130-src
