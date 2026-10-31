"""Saved activations wait in pinned host memory between the forward and the backward pass.

`SOLUTION_OFFLOAD=1`. Everything autograd keeps for the backward pass of the decoder layers (the
fp32 residual inputs of the norms, the fp16 inputs of the projections, attention's q/k/v and
output, the dropout masks) is copied to a pinned host arena right after it is produced, on a
separate CUDA stream, and the GPU copy is released. The backward pass asks for them in roughly
the reverse order: each request also starts the copies of the next few tensors back to the GPU,
so the bus works while the GPU computes. The tensors are the same bytes; nothing is recomputed
or approximated, and nothing is cached between runs. This is our own pack/unpack pair on
`torch.autograd.graph.saved_tensors_hooks`, not Soup's `activation_offloading` switch (which
is left as the base run has it).
"""

import os
import threading

import torch


class Arena:
    """One pinned host buffer, handed out in order and reset once per step."""

    def __init__(self, nbytes):
        self.buf = torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)
        self.size = nbytes
        self.top = 0

    def take(self, nbytes):
        start = (self.top + 255) // 256 * 256
        if start + nbytes > self.size:
            return None
        self.top = start + nbytes
        return self.buf[start:start + nbytes]


class Record:
    __slots__ = ("host", "shape", "dtype", "device", "event", "gpu", "index", "src")

    def __init__(self, host, shape, dtype, device, event, index):
        self.host, self.shape, self.dtype, self.device = host, shape, dtype, device
        self.event, self.gpu, self.index, self.src = event, None, index, None


class Offloader:
    def __init__(self, arena_mb=2048, min_bytes=1 << 20, ahead=8, ring_mb=192):
        self.arena = None
        self._arena_thread = threading.Thread(target=self._make_arena, args=(arena_mb << 20,), daemon=True)
        self._arena_thread.start()  # pinning 2 GB takes a while: it overlaps the model's load
        self.min_bytes = min_bytes
        self.ahead = ahead
        self.stream = torch.cuda.Stream()
        self.records = []
        self.pending = []  # packed, copy not issued yet
        self.batch = int(os.environ.get("SOLUTION_OFFLOAD_BATCH", "1"))  # measured: later groups hold GPU memory longer
        self.inflight = []  # prefetched records not yet handed to autograd, oldest first
        self.lock = threading.Lock()
        self.bytes_step = 0
        self.kept = 0

    def _make_arena(self, nbytes):
        self.arena = Arena(nbytes)

    def new_step(self):
        """The previous backward pass is over: the arena starts again from the top."""
        if self._arena_thread is not None:
            self._arena_thread.join()
            self._arena_thread = None
        if os.environ.get("SOLUTION_TIMELINE") == "1" and self.records:
            print(f"[solution-tl] offload step: {len(self.records)} tensors, {self.bytes_step >> 20} MiB"
                  f" to host, {self.kept >> 20} MiB kept", file=__import__("sys").stderr, flush=True)
        with self.lock:
            # a guess of the last backward pass that was never asked for may still be arriving:
            # its buffer goes back to the pool only after the copy, in stream order
            current = torch.cuda.current_stream()
            for rec in self.records:
                if rec.gpu is not None and rec.event is not None:
                    current.wait_event(rec.event)
                    rec.gpu = None
            self.arena.top = 0
            self.records = []
            self.pending = []
            self.inflight = []
            self.bytes_step = 0
            self.kept = 0

    def pack(self, t):
        if not t.is_cuda or t.dtype not in (torch.float16, torch.float32, torch.bfloat16) \
                or t.numel() * t.element_size() < self.min_bytes:
            return t
        # a strided view (attention's per-sequence slices, the rotated q/k) is copied as a dense
        # tensor and comes back dense: the backward functions take any layout
        nbytes = t.numel() * t.element_size()
        with self.lock:
            host = self.arena.take(nbytes) if self.arena is not None else None
            if host is None:
                self.kept += nbytes
                return t  # the arena is full: this one stays on the GPU
            rec = Record(host, t.shape, t.dtype, t.device, None, len(self.records))
            rec.src = t  # the GPU tensor lives until its copy is issued and done
            self.records.append(rec)
            self.pending.append(rec)
            self.bytes_step += nbytes
            if len(self.pending) >= self.batch:
                self._flush()
            return rec

    def _flush(self):
        """Issue the queued host copies as one group: one wait on the compute stream, one event."""
        pending, self.pending = self.pending, []
        if not pending:
            return
        current = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(current)  # every queued tensor is complete on the compute stream
            for rec in pending:
                rec.host.view(rec.dtype).view(rec.shape).copy_(rec.src, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        for rec in pending:
            rec.src.record_stream(self.stream)  # the GPU bytes live until the copy has read them
            rec.src = None
            rec.event = event

    def _fetch_group(self, recs):
        """Copies back for `recs` (those not on the GPU yet) as one group: one wait, one event.
        The buffers come from the compute stream's pool (one pool, one high-water mark); the copy
        stream first waits for everything queued on the compute stream, so a block freed there a
        moment ago is never written while the kernel that read it may still be running."""
        recs = [r for r in recs if r.gpu is None and r.src is None]
        if not recs:
            return
        for rec in recs:  # allocated before the wait: a block freed later on the compute stream
            rec.gpu = torch.empty(rec.shape, dtype=rec.dtype, device=rec.device)  # cannot be one of these
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):  # after the records' own D2H copies: same stream
            for rec in recs:
                rec.gpu.copy_(rec.host.view(rec.dtype).view(rec.shape), non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        for rec in recs:
            rec.event = event
            self.inflight.append(rec)
        while len(self.inflight) > 2 * self.ahead:  # a guess that was never asked for: let it go
            old = self.inflight.pop(0)
            if old.gpu is not None:
                torch.cuda.current_stream().wait_event(old.event)  # freed only after its copy landed
                old.gpu = None

    def unpack(self, packed):
        if not isinstance(packed, Record):
            return packed
        with self.lock:
            if self.pending:
                self._flush()  # the backward has begun: every host copy is on its way
            group = [packed] + [self.records[j] for j in range(packed.index - 1,
                                                                 max(-1, packed.index - 1 - self.ahead), -1)]
            self._fetch_group(group)  # the backward walks the records backwards
            if packed in self.inflight:
                self.inflight.remove(packed)  # asked for: it is not a guess any more
            torch.cuda.current_stream().wait_event(packed.event)
            gpu = packed.gpu
            packed.gpu = None  # autograd keeps the returned tensor as long as it needs it
            return gpu


ACTIVE = None


def start():
    global ACTIVE
    from . import setting

    if setting("SOLUTION_OFFLOAD") != "1" or not torch.cuda.is_available():
        return None
    ACTIVE = Offloader(arena_mb=int(os.environ.get("SOLUTION_OFFLOAD_ARENA_MB", "2048")),
                       min_bytes=int(float(os.environ.get("SOLUTION_OFFLOAD_MIN_MB", "1")) * (1 << 20)),
                       ahead=int(os.environ.get("SOLUTION_OFFLOAD_AHEAD", "8")),
                       ring_mb=int(os.environ.get("SOLUTION_OFFLOAD_RING_MB", "128")))
    return ACTIVE
