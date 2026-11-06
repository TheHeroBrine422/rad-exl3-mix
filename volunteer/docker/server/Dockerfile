# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# The rad-quant@home queue server image (radquant-server): server.py + admin.py (+ the CPU checker), Python stdlib +
# numpy, no GPU, non-root.
# Build with volunteer/docker/server/build.sh (a minimal context of the needed paths: never local/ or data).
# Run with compose.yaml next to this file (README "Server container"); config by RADQUANT_* environment variables
# (the v0 names RXMQ_* are still read in 0.2).
ARG PY_IMAGE=python:3.14-slim-trixie
FROM ${PY_IMAGE}

ARG VERSION=0.0.0-dev
ARG REPO_COMMIT=unknown
ARG SOURCE=https://github.com/TheHeroBrine422/rad-exl3-mix
ARG CREATED=unknown
ARG NUMPY_SPEC="numpy==2.5.4"

# numpy only for `radquant admin check` (the quant checker); the server itself is stdlib
RUN pip install --no-cache-dir --root-user-action=ignore "${NUMPY_SPEC}" \
 && groupadd --system --gid 10001 radquant \
 && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin radquant

COPY LICENSE NOTICE /opt/radquant/
COPY volunteer/common /opt/radquant/volunteer/common
COPY volunteer/jobs /opt/radquant/volunteer/jobs
COPY volunteer/server /opt/radquant/volunteer/server
COPY volunteer/checker /opt/radquant/volunteer/checker
COPY volunteer/docker/server/entrypoint.sh /usr/local/bin/radquant
COPY volunteer/docker/server/healthcheck.py /opt/radquant/healthcheck.py

# db: the SQLite file (+ -wal/-shm); out: results/ and spool/ on ONE filesystem (completion renames spool -> results);
# data: the per-file inputs (read-only for the server); sources: what datapack symlinks into data (bf16, Hessians)
RUN python -m compileall -q /opt/radquant \
 && chmod 0755 /usr/local/bin/radquant && ln -s radquant /usr/local/bin/rxmq && chmod -R a-w /opt/radquant \
 && mkdir -p /srv/radquant/db /srv/radquant/data /srv/radquant/out/results /srv/radquant/out/spool /srv/radquant/sources \
 && chown -R 10001:10001 /srv/radquant/db /srv/radquant/data /srv/radquant/out \
 && echo "radquant-server ${VERSION} repo ${REPO_COMMIT}" > /opt/radquant/IMAGE

ENV RADQUANT_DB=/srv/radquant/db/queue.sqlite RADQUANT_DATA=/srv/radquant/data RADQUANT_RESULTS=/srv/radquant/out/results \
    RADQUANT_SPOOL=/srv/radquant/out/spool RADQUANT_BIND=0.0.0.0 RADQUANT_PORT=8740 PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.title="radquant-server" \
      org.opencontainers.image.description="rad-quant@home: job queue server for donated-compute EXL3 quantization and evals (rad-exl3-mix volunteer/)" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.url="${SOURCE}/tree/main/volunteer" \
      org.opencontainers.image.documentation="${SOURCE}/blob/main/volunteer/README.md" \
      org.opencontainers.image.revision="${REPO_COMMIT}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.licenses="Apache-2.0"
USER 10001:10001
WORKDIR /srv/radquant
EXPOSE 8740
VOLUME ["/srv/radquant/db", "/srv/radquant/out"]
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 CMD ["python", "-I", "/opt/radquant/healthcheck.py"]
ENTRYPOINT ["/usr/local/bin/radquant"]
CMD ["serve"]
