#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p2p231.py MODE SECONDS -- w231: stress the cross-card traffic of qad220's two-card pipeline (run_pipe) without the model, to see whether the
sporadic "illegal memory access" comes from the transfer path. Two visible cards A = cuda:0, B = cuda:1; the main thread plays card A and a worker
thread card B, as run_pipe does: A computes and sends a 10 MB hidden state [1024, 5120] bf16 to B, B computes and sends a gradient of the same size
back; both cards also allocate / free buffers of deep-run sizes (64 MB .. 2 GB) so the allocator maps and unmaps memory while copies run.
MODE p2p  = run_pipe's transfers (x.to(other card): a peer copy on the source card's stream, issued by the receiving thread for A -> B)
MODE host = through pinned host buffers (D2H + event on the source card, H2D on the destination; each card touched by its own thread only)
Prints a line every 30 s and the totals; a CUDA error ends the run with exit 1 and the traceback."""
import sys, time, threading, queue
import torch

mode, secs = sys.argv[1], float(sys.argv[2])
assert mode in ("p2p", "host")
A, B = torch.device("cuda:0"), torch.device("cuda:1")
print(torch.__version__, torch.cuda.get_device_name(0), "| peer", torch.cuda.can_device_access_peer(0, 1), "| mode", mode, flush=True)
H = (1024, 5120)
stop = time.time() + secs
qb, qg = queue.Queue(), queue.Queue()
err = []


def work(dev, w, x):
    """a few GEMMs on dev plus allocator churn at deep-run sizes"""
    for _ in range(4): x = torch.tanh(x @ w)
    big = torch.empty(int((64 << 20) * (1 + torch.randint(0, 32, ()).item())), dtype=torch.uint8, device=dev); big.fill_(1)
    return x, big


def send(x):
    """the host path: x into a pinned buffer (async on x's card), with the event the receiver waits on"""
    h = torch.empty(x.shape, dtype=x.dtype, pin_memory=True); h.copy_(x, non_blocking=True)
    ev = torch.cuda.Event(); ev.record(); return (h, ev)


def recv(m, dev):
    if mode == "p2p": return m.to(dev)
    h, ev = m; ev.synchronize(); return h.to(dev, non_blocking=True)


def side_b():
    try:
        torch.cuda.set_device(B)
        w = torch.randn(5120, 5120, device=B, dtype=torch.bfloat16) / 80
        while True:
            m = qb.get()
            if m is None: return
            j, msg = m
            x = recv(msg, B)                                       # p2p: hA.to(B) in the worker (a copy on A's stream, as run_pipe)
            y, big = work(B, w, x)
            float(y[0, 0])                                         # run_pipe's float(kl): a sync on B
            g = y.contiguous()
            qg.put((j, g if mode == "p2p" else send(g)))
            del big, y, x
    except BaseException as e:
        err.append(e); qg.put((None, None))


th = threading.Thread(target=side_b, daemon=True); th.start()
n = 0; t_last = time.time(); pend = []
try:
    with torch.cuda.device(A):
        w = torch.randn(5120, 5120, device=A, dtype=torch.bfloat16) / 80
        while time.time() < stop and not err:
            x = torch.randn(H, device=A, dtype=torch.bfloat16)
            h, big = work(A, w, x)
            qb.put((n, h if mode == "p2p" else send(h))); pend.append(n); n += 1
            if len(pend) > 2:
                j, g = qg.get()
                if j is None: break
                gA = recv(g, A)                                    # p2p: gx.to(A) in the main thread (a copy on B's stream)
                (gA.float().sum() + h.float().sum()).item(); pend.pop(0)
            del big
            if time.time() - t_last > 30: print(f"{time.strftime('%H:%M:%S')} {n} round trips", flush=True); t_last = time.time()
        torch.cuda.synchronize(A); torch.cuda.synchronize(B)
finally:
    qb.put(None); th.join(timeout=60)
if err: print("ERROR in card-B thread:", repr(err[0]), flush=True); sys.exit(1)
print(f"done: {n} round trips in {secs:.0f} s, no error", flush=True)
