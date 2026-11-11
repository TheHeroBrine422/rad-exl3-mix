# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rxrun: the local ladder runner (w105, 2026-11-09). Runs rxq quant jobs on our own cards, faster than one
`python -m rxq job` process a job, with outputs byte-identical to it:

- worker.py  a persistent job process: `rxq job` semantics (rxq.job.run_quant, same arguments), one engine setup and
             one kernel load for many jobs; specs on stdin, one JSON reply a job on stdout
- pool.py    N jobs at once on one card (persistent workers or one process a job), VRAM / host-RAM aware, the card lock
             held (ram.lock shared) for a session, a hold file, input prefetch into the page cache, a host-RAM guard
- run.py     the restartable ladder driver: pass 1 -> 2 -> 3 (rxq.ladder.levels), per level specs -> jobs -> assemble
             -> swj level pass -> post apply -> pack recipe; status line with an ETA from the measured rate
- cmp.py     byte-identity of two job / level dirs (tensor bytes + metadata; --raw whole files; --tensors across stamps)
- seedsd.py  the proxy's seed SD per family and K from run.py's duplicate-seed jobs

Lives outside rxq/ on purpose: rxq's code stamp hashes rxq/*.py, so the runner does not move the stamp of the
levels it makes (a level made by rxrun carries the same stamps as one made by `rxq job` processes)."""
