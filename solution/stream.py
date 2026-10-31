"""The 4-bit weights live in pinned host memory and reach the GPU one layer ahead of their use.

`SOLUTION_STREAM=1`. The frozen NF4 weights of the 28 decoder layers (about 650 MB packed) are
read exactly once per forward and once per backward of their layer (twice for the layers that
rebuild their MLP activations), always in the same order. After Soup has built the model, every
`QuantWeight` hands its packed bytes and absmax codes to pinned host memory and gives the GPU
copy back. The first training step records the order in which the weights are touched; from the
second step on, each access also starts the copies of the next few weights on a separate CUDA
stream, so a layer's bytes are on the GPU before its first matmul and leave right after its
last one. Nothing is cached between runs and the arithmetic is unchanged: the same bytes are
unpacked by the same kernel; only where they wait between uses differs.
"""

import os
import threading

import torch


class Ring:
    """One GPU buffer handed out region by region, like a ring. A region is written again only
    after the event recorded when its previous occupant was last read has passed on the copy
    stream; a region whose occupant is still in use is never handed out (the caller then takes
    an ordinary buffer). The staging memory of a streamer is therefore a fixed amount,
    allocated once, outside the caching allocator's per-stream pools."""

    def __init__(self, nbytes, device, copy_stream):
        self.buf = torch.empty((nbytes,), dtype=torch.uint8, device=device)
        self.base = self.buf.data_ptr()
        self.size = nbytes
        self.top = 0
        self.copy_stream = copy_stream
        self.regions = {}  # start -> [end, event or None (still in use)]
        self.lock = threading.Lock()

    def take(self, nbytes, dtype, shape):
        """A view for `nbytes` bytes, or None when the ring cannot give one right now."""
        need = (nbytes + 255) // 256 * 256
        if need > self.size:
            return None
        with self.lock:
            start = self.top if self.top + need <= self.size else 0
            end = start + need
            waits = []
            for s0, (e0, event) in self.regions.items():
                if s0 < end and e0 > start:  # overlaps the region we want
                    if event is None:
                        return None  # still being read: the ring is full here
                    waits.append(s0)
            for s0 in waits:
                self.copy_stream.wait_event(self.regions.pop(s0)[1])
            self.regions[start] = [end, None]
            self.top = end
        return self.buf[start:start + nbytes].view(dtype).view(shape)

    def release(self, start_ptr, stream):
        """The region at `start_ptr` may be rewritten once everything queued on `stream` ran."""
        start = start_ptr - self.base
        event = torch.cuda.Event()
        event.record(stream)
        with self.lock:
            region = self.regions.get(start)
            if region is not None:
                region[1] = event

    def holds(self, tensor):
        return self.base <= tensor.data_ptr() < self.base + self.size


class Streamer:
    def __init__(self, ahead=6, ring_mb=64):
        self.ahead = ahead
        self.stream = torch.cuda.Stream()
        self.ring_mb = ring_mb
        self.ring = None
        self.order = []  # QuantWeights in the order the first step touched them
        self.recording = True
        self.pos = 0
        self.lock = threading.Lock()

    def register(self, qw):
        """Move `qw`'s packed bytes and absmax codes to pinned host memory; free the GPU copies
        (a weight already taken to the host at load time, see `patch_load`, is only pinned)."""
        qw.packed_host = qw.packed.detach().to("cpu").pin_memory() if qw.packed.is_cuda else qw.packed.pin_memory()
        qw.absmax_host = qw.absmax.detach().to("cpu").pin_memory() if qw.absmax.is_cuda else qw.absmax.pin_memory()
        qw.packed = qw.absmax = None
        qw.gpu = None
        qw.event = None
        qw.streamer = self

    def _issue(self, qw):
        if qw.gpu is not None:
            return
        if self.ring is None:
            self.ring = Ring(self.ring_mb << 20, qw.device, self.stream)
        packed = self.ring.take(qw.packed_host.numel(), qw.packed_host.dtype, qw.packed_host.shape)
        absmax = self.ring.take(qw.absmax_host.numel(), qw.absmax_host.dtype, qw.absmax_host.shape)
        if packed is None or absmax is None:  # the ring is full: ordinary buffers, written after the queue drained
            packed = torch.empty_like(qw.packed_host, device=qw.device) if packed is None else packed
            absmax = torch.empty_like(qw.absmax_host, device=qw.device) if absmax is None else absmax
            self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            packed.copy_(qw.packed_host, non_blocking=True)
            absmax.copy_(qw.absmax_host, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        qw.gpu, qw.event = (packed, absmax), event
        if os.environ.get("SOLUTION_STREAM_SYNC") == "1":  # debugging aid: no overlap at all
            torch.cuda.synchronize()

    def fetch(self, qw):
        """The GPU copies of `qw` (ready on the current stream); prefetch what comes next."""
        with self.lock:
            n = len(self.order)
            if self.recording:
                if n and qw is self.order[0]:
                    self.recording = False  # the sequence starts again: one whole step is known
                    self.pos = 1
                else:
                    self.order.append(qw)
            else:
                if self.order[self.pos % n] is not qw:
                    for k in range(n):  # the order changed: find this weight further on
                        if self.order[(self.pos + k) % n] is qw:
                            self.pos += k
                            break
                self.pos += 1
            self._issue(qw)
            if not self.recording:
                for k in range(self.ahead):
                    self._issue(self.order[(self.pos + k) % n])
            torch.cuda.current_stream().wait_event(qw.event)
            return qw.gpu

    def release(self, qw, packed, absmax):
        """After the dequantization kernel was launched: the buffers go back once it is done."""
        if os.environ.get("SOLUTION_STREAM_SYNC") == "1":
            torch.cuda.synchronize()
        stream = torch.cuda.current_stream()
        for t in (packed, absmax):
            if self.ring.holds(t):
                self.ring.release(t.data_ptr(), stream)
        qw.gpu = qw.event = None


def wanted():
    from . import setting

    return setting("SOLUTION_STREAM") == "1"


def offload_model(model):
    """Build the QuantWeights of every decoder layer now, so their bytes move to the host before
    the first step instead of staying on the GPU until each layer's first use."""
    from . import quant
    from .attention import merged_qkv
    from .fused_mlp import merged_gu

    if quant.STREAMER is None:
        return 0
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        base = getattr(model, "base_model", None)
        inner = getattr(base, "model", None)
        layers = getattr(getattr(inner, "model", None), "layers", None)
    if layers is None:
        for name, module in model.named_modules():
            if name.endswith("model.layers"):
                layers = module
                break
    if layers is None:
        return 0
    count = 0
    for layer in layers:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        mods = [getattr(attn, n, None) for n in ("q_proj", "k_proj", "v_proj")]
        bases = [getattr(m, "base_layer", m) for m in mods]
        o = getattr(attn, "o_proj", None)
        o = getattr(o, "base_layer", o)
        gate, up, down = (getattr(mlp, n, None) for n in ("gate_proj", "up_proj", "down_proj"))
        try:
            if all(getattr(b, "weight", None) is not None and quant.supported(b.weight.quant_state)
                   and b.bias is not None for b in bases):
                merged_qkv(attn, bases)
                count += 1
            if o is not None and quant.supported(o.weight.quant_state):
                quant.quant_weight_of(o)
                count += 1
            if all(m is not None and quant.supported(m.weight.quant_state) for m in (gate, up, down)):
                merged_gu(mlp)
                quant.quant_weight_of(down)
                count += 2
        except Exception as exc:  # noqa: BLE001 - a layer we do not understand stays as it is
            print(f"solution: layer not streamed ({exc!r})")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return count


def patch_load():
    """`SOLUTION_STREAM=1` with the weights taken to the host as the model loads: bitsandbytes
    quantizes each 4-bit weight on the GPU inside `Params4bit._quantize`; right after, its packed
    bytes and absmax codes go to the host and the GPU keeps a zero-stride placeholder, so the
    model's load never holds all 4-bit weights on the GPU at once. The host copies travel on the
    quant_state (bitsandbytes rebuilds the parameter object in `to()`, the state is shared)."""
    try:
        from bitsandbytes.nn import modules
        import torch
    except Exception:  # noqa: BLE001
        return False
    from . import quant

    cls = modules.Params4bit
    original = cls._quantize

    counter = [0]

    def _quantize(self, device):
        result = original(self, device)
        counter[0] += 1
        if os.environ.get("SOLUTION_TIMELINE") == "1" and counter[0] in (1, 2, 50, 100, 150, 196):
            from .startup import mark

            mark(f"quantized-{counter[0]} {tuple(result.quant_state.shape) if getattr(result, 'quant_state', None) else ''}")
        try:
            state = getattr(result, "quant_state", None)
            if quant.STREAMER is not None and result.is_cuda and quant.supported(state) \
                    and getattr(state, "_solution_host", None) is None:
                packed = result.data.detach().to("cpu").view(-1)
                absmax = state.absmax.detach().to("cpu").view(-1)
                state._solution_host = (packed, absmax)
                result.data = torch.zeros((1, 1), dtype=result.data.dtype, device=device).expand_as(result.data)
                state.absmax = torch.zeros((1,), dtype=state.absmax.dtype, device=device).expand_as(state.absmax)
        except Exception as exc:  # noqa: BLE001 - a weight we do not understand stays on the GPU
            print(f"solution: weight kept on the GPU at load ({exc!r})")
        return result

    cls._quantize = _quantize
    return True
