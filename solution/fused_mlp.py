"""The Qwen2 MLP as one merged gate|up 4-bit matmul, a 2-D SwiGLU kernel over that buffer, and the
down projection; optionally with the gate|up activations recomputed in the backward pass.

Base Soup runs gate_proj and up_proj as two separate 4-bit linears (two unpackings, two GEMMs
reading the same input), then silu and a product as separate ops. Here the two packed weights are
stacked once into one [2I, K] NF4 weight (solution/quant.py): one unpacking, one GEMM with twice
the width, and `swiglu_gated_kernel` reads gate and up as the two halves of each row of that
buffer. Its backward writes the gradient of both halves into one [R, 2I] buffer, which is exactly
what the merged GEMM's backward consumes.

With `recompute=True` the [R, 2I] gate|up buffer is not kept for the backward pass (that is the
largest activation of the model: 28 layers x 4 x 512 x 17,920 x 2 bytes = 2 GB); it is rebuilt
from the saved input with one extra GEMM per layer in the backward pass. Our own code, not Soup's
`gradient_checkpointing` switch.
"""

import torch
import triton
import triton.language as tl

ROWS, COLS = 32, 128


@triton.jit(do_not_specialize=["R", "I", "stride_gu", "stride_h"])
def swiglu_gated_kernel(gu_ptr, h_ptr, R, I, stride_gu, stride_h, ROWS: tl.constexpr,
                        COLS: tl.constexpr):
    """h[r, c] = silu(gu[r, c]) * gu[r, I + c] on a ROWS x COLS tile."""
    r0 = tl.program_id(0) * ROWS
    c0 = tl.program_id(1) * COLS
    rows = r0 + tl.arange(0, ROWS)[:, None]
    cols = c0 + tl.arange(0, COLS)[None, :]
    inside = (rows < R) & (cols < I)
    g = tl.load(gu_ptr + rows * stride_gu + cols, mask=inside, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + rows * stride_gu + I + cols, mask=inside, other=0.0).to(tl.float32)
    h = u * (g / (1.0 + tl.exp(-g)))
    tl.store(h_ptr + rows * stride_h + cols, h.to(h_ptr.dtype.element_ty), mask=inside)


@triton.jit(do_not_specialize=["R", "I", "stride_dh", "stride_gu", "stride_dgu"])
def swiglu_gated_bwd_kernel(dh_ptr, gu_ptr, dgu_ptr, R, I, stride_dh, stride_gu, stride_dgu,
                            ROWS: tl.constexpr, COLS: tl.constexpr):
    """dgu[r, c] = dh * u * d silu(g) / dg and dgu[r, I + c] = dh * silu(g), on a tile."""
    r0 = tl.program_id(0) * ROWS
    c0 = tl.program_id(1) * COLS
    rows = r0 + tl.arange(0, ROWS)[:, None]
    cols = c0 + tl.arange(0, COLS)[None, :]
    inside = (rows < R) & (cols < I)
    dh = tl.load(dh_ptr + rows * stride_dh + cols, mask=inside, other=0.0).to(tl.float32)
    g = tl.load(gu_ptr + rows * stride_gu + cols, mask=inside, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + rows * stride_gu + I + cols, mask=inside, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    silu = g * s
    dsilu = s + silu * (1.0 - s)
    dg = dh * u * dsilu
    du = dh * silu
    tl.store(dgu_ptr + rows * stride_dgu + cols, dg.to(dgu_ptr.dtype.element_ty), mask=inside)
    tl.store(dgu_ptr + rows * stride_dgu + I + cols, du.to(dgu_ptr.dtype.element_ty), mask=inside)


def swiglu_forward(gu):
    """gu: [..., 2I] contiguous -> h: [..., I]."""
    I = gu.shape[-1] // 2
    gu2 = gu.reshape(-1, 2 * I)
    R = gu2.shape[0]
    h = torch.empty((R, I), dtype=gu.dtype, device=gu.device)
    grid = (triton.cdiv(R, ROWS), triton.cdiv(I, COLS))
    swiglu_gated_kernel[grid](gu2, h, R, I, gu2.stride(0), h.stride(0), ROWS=ROWS, COLS=COLS,
                              num_warps=4)
    return h.view(*gu.shape[:-1], I)


def swiglu_backward(dh, gu):
    I = gu.shape[-1] // 2
    gu2 = gu.reshape(-1, 2 * I)
    dh2 = dh.reshape(-1, I)
    if dh2.stride(-1) != 1:
        dh2 = dh2.contiguous()
    R = gu2.shape[0]
    dgu = torch.empty_like(gu2)
    grid = (triton.cdiv(R, ROWS), triton.cdiv(I, COLS))
    swiglu_gated_bwd_kernel[grid](dh2, gu2, dgu, R, I, dh2.stride(0), gu2.stride(0), dgu.stride(0),
                                  ROWS=ROWS, COLS=COLS, num_warps=4)
    return dgu.view_as(gu)


class SwiGLUGated(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        gu = gu.contiguous()
        ctx.save_for_backward(gu)
        return swiglu_forward(gu)

    @staticmethod
    def backward(ctx, dh):
        (gu,) = ctx.saved_tensors
        return swiglu_backward(dh.to(gu.dtype), gu)


class MLPRecompute(torch.autograd.Function):
    """down(swiglu(x @ Wgu^T)) @ ... with nothing but x kept for the backward pass."""

    @staticmethod
    def forward(ctx, x, qw_gu, qw_d):
        with torch.no_grad():
            gu = torch.matmul(x, qw_gu.dequantize(x.dtype).t())
            h = swiglu_forward(gu)
            del gu
            out = torch.matmul(h, qw_d.dequantize(x.dtype).t())
        ctx.save_for_backward(x)
        ctx.qw_gu, ctx.qw_d = qw_gu, qw_d
        return out

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        dout = dout.to(x.dtype)
        w_gu = ctx.qw_gu.dequantize(x.dtype)
        gu = torch.matmul(x, w_gu.t())
        dh = torch.matmul(dout, ctx.qw_d.dequantize(x.dtype))
        dgu = swiglu_backward(dh, gu)
        del dh, gu
        dx = torch.matmul(dgu, w_gu)
        return dx, None, None


_LAYER_OF = {}


def _wants_recompute(mlp, count):
    """True for the MLPs of the last `count` decoder layers (`layer_idx` of their attention)."""
    if not count:
        return False
    idx, total = _LAYER_OF.get(id(mlp), (None, None))
    if idx is None:
        return True  # layer unknown: recompute (the "all layers" behaviour)
    return idx >= total - count


def register_layers(model):
    """Record which decoder layer each MLP belongs to (called once the model is built)."""
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        base = getattr(model, "get_base_model", None)
        if callable(base):
            layers = getattr(getattr(base(), "model", None), "layers", None)
    if layers is None:
        return 0
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            _LAYER_OF[id(mlp)] = (i, len(layers))
    return len(layers)


def merged_gu(mlp):
    """The cached merged gate|up QuantWeight of one MLP module."""
    from . import quant

    gate, up = getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None)
    key = (id(gate.weight), id(up.weight))
    qw = getattr(mlp, "_solution_gu", None)
    if qw is None or getattr(mlp, "_solution_gu_key", None) != key:
        qw = quant.QuantWeight([gate.weight, up.weight], release=True)
        mlp._solution_gu, mlp._solution_gu_key = qw, key
    return qw


def patch_qwen2_mlp(dtype=torch.float16, recompute=0):
    """`recompute`: how many decoder layers (counted from the last one) rebuild their gate|up
    activations in the backward pass instead of keeping them; 0 keeps everything."""
    from transformers.models.qwen2 import modeling_qwen2

    from . import quant

    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:  # noqa: BLE001
        Linear4bit = ()  # noqa: N806 - no bitsandbytes: the patch always falls through

    original = modeling_qwen2.Qwen2MLP.forward

    merged = merged_gu

    def forward(self, x):
        if not hasattr(self, "_solution_recompute"):
            self._solution_recompute = _wants_recompute(self, recompute)
        gate, up, down = (getattr(self, n, None) for n in ("gate_proj", "up_proj", "down_proj"))
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda \
                or not all(isinstance(m, Linear4bit) and m.bias is None for m in (gate, up, down)) \
                or not all(quant.supported(m.weight.quant_state) for m in (gate, up, down)):
            return original(self, x)
        x16 = x if x.dtype == dtype else x.to(dtype)
        qw_gu, qw_d = merged(self), quant.quant_weight_of(down)
        if recompute and torch.is_grad_enabled() and self._solution_recompute:
            return MLPRecompute.apply(x16, qw_gu, qw_d)
        gu = quant.quant_linear(x16, qw_gu)
        h = SwiGLUGated.apply(gu)
        return quant.quant_linear(h, qw_d)

    modeling_qwen2.Qwen2MLP.forward = forward
