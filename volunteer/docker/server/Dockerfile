# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# The queue server image: server.py + admin.py (+ the CPU checker), Python stdlib + numpy, no GPU, non-root.
# Build with volunteer/docker/server/build.sh (a minimal context of the needed paths: never local/ or data).
# Run with compose.yaml next to this file (README "Server container"); config by RXMQ_* environment variables.
ARG PY_IMAGE=python:3.14-slim-trixie
FROM ${PY_IMAGE}

ARG VERSION=0.0.0-dev
ARG REPO_COMMIT=unknown
ARG SOURCE=https://github.com/TheHeroBrine422/rad-exl3-mix
ARG CREATED=unknown
ARG NUMPY_SPEC="numpy==2.5.4"

# numpy only for `rxmq admin check` (the checker); the server itself is stdlib
RUN pip install --no-cache-dir --root-user-action=ignore "${NUMPY_SPEC}" \
 && groupadd --system --gid 10001 rxmq \
 && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin rxmq

COPY LICENSE NOTICE /opt/rxmq/
COPY volunteer/common /opt/rxmq/volunteer/common
COPY volunteer/jobs /opt/rxmq/volunteer/jobs
COPY volunteer/server /opt/rxmq/volunteer/server
COPY volunteer/checker /opt/rxmq/volunteer/checker
COPY volunteer/docker/server/entrypoint.sh /usr/local/bin/rxmq
COPY volunteer/docker/server/healthcheck.py /opt/rxmq/healthcheck.py

# db: the SQLite file (+ -wal/-shm); out: results/ and spool/ on ONE filesystem (completion renames spool -> results);
# data: the per-file inputs (read-only for the server); sources: what datapack symlinks into data (bf16, Hessians)
RUN python -m compileall -q /opt/rxmq \
 && chmod 0755 /usr/local/bin/rxmq && chmod -R a-w /opt/rxmq \
 && mkdir -p /srv/rxmq/db /srv/rxmq/data /srv/rxmq/out/results /srv/rxmq/out/spool /srv/rxmq/sources \
 && chown -R 10001:10001 /srv/rxmq/db /srv/rxmq/data /srv/rxmq/out \
 && echo "rxm-queue ${VERSION} repo ${REPO_COMMIT}" > /opt/rxmq/IMAGE

ENV RXMQ_DB=/srv/rxmq/db/queue.sqlite RXMQ_DATA=/srv/rxmq/data RXMQ_RESULTS=/srv/rxmq/out/results \
    RXMQ_SPOOL=/srv/rxmq/out/spool RXMQ_BIND=0.0.0.0 RXMQ_PORT=8740 PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.title="rxm-queue" \
      org.opencontainers.image.description="Job queue server for donated-compute EXL3 quantization (rad-exl3-mix volunteer/)" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.url="${SOURCE}/tree/main/volunteer" \
      org.opencontainers.image.documentation="${SOURCE}/blob/main/volunteer/README.md" \
      org.opencontainers.image.revision="${REPO_COMMIT}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.licenses="Apache-2.0"
USER 10001:10001
WORKDIR /srv/rxmq
EXPOSE 8740
VOLUME ["/srv/rxmq/db", "/srv/rxmq/out"]
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 CMD ["python", "-I", "/opt/rxmq/healthcheck.py"]
ENTRYPOINT ["/usr/local/bin/rxmq"]
CMD ["serve"]
