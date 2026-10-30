"""NF4 weights unpacked by one Triton kernel, and the 4-bit linear around it.

bitsandbytes stores a 4-bit weight as packed nibbles (two per byte), one absmax per block of 64
values, and (double quantization) those absmax values themselves as uint8 codes with one fp32
scale per 256 of them plus an offset. For a batch this large (4 x 512 rows) bitsandbytes does not
run its fused GEMM: every projection first launches a kernel to rebuild the absmax values, then a
kernel to unpack the weight into a dense fp16 matrix, then cuBLAS; the backward pass unpacks the
same weight again for the gradient of the input. That is 28 layers x 7 projections x 2 passes x 2
kernels per step.

`nf4_dequant_kernel` does both levels in one pass: each program takes a run of packed bytes, looks
up the absmax code of its block, applies the second-level scale and offset, and writes the dense
rows in the compute dtype. `QuantLinear` is the autograd function around it (the weight is frozen,
so only the input needs a gradient), used from `solution/lowprec.py` in place of `bnb.matmul_4bit`.
"""

import torch
import triton
import triton.language as tl

NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
       0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_u8_ptr, code_ptr, scale2_ptr, offset, nf4_ptr, out_ptr,
                       n_bytes, BLOCK: tl.constexpr, BLOCK2: tl.constexpr, BYTES: tl.constexpr):
    """BYTES packed bytes per program (2 * BYTES values, a whole number of BLOCK-value blocks)."""
    pid = tl.program_id(0)
    offs = pid * BYTES + tl.arange(0, BYTES)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    hi = byte >> 4
    lo = byte & 15
    # value index i = 2 * byte index (hi) and 2 * byte index + 1 (lo); its block is i // BLOCK
    block = (offs * 2) // BLOCK
    code = tl.load(absmax_u8_ptr + block, mask=mask, other=0).to(tl.int32)
    scale2 = tl.load(scale2_ptr + block // BLOCK2, mask=mask, other=0.0)
    absmax = tl.load(code_ptr + code, mask=mask, other=0.0) * scale2 + offset
    v_hi = tl.load(nf4_ptr + hi, mask=mask, other=0.0) * absmax
    v_lo = tl.load(nf4_ptr + lo, mask=mask, other=0.0) * absmax
    # interleave (hi, lo) pairs into one contiguous run of 2 * BYTES values: one coalesced store
    v = tl.reshape(tl.join(v_hi, v_lo), [BYTES * 2])
    out_offs = pid * BYTES * 2 + tl.arange(0, BYTES * 2)
    tl.store(out_ptr + out_offs, v.to(out_ptr.dtype.element_ty), mask=out_offs < n_bytes * 2)


_TABLES = {}


def _nf4_table(device):
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(NF4, dtype=torch.float32, device=device)
    return _TABLES[key]


def supported(quant_state):
    """The layout this kernel handles: NF4, blocks of 64, double-quantized absmax in blocks of 256."""
    return quant_state is not None and getattr(quant_state, "quant_type", None) == "nf4" \
        and getattr(quant_state, "nested", False) and quant_state.blocksize == 64 \
        and getattr(quant_state.state2, "blocksize", None) == 256 \
        and quant_state.absmax.dtype == torch.uint8 and quant_state.state2.code.numel() == 256


def dequantize_nf4(packed, quant_state, dtype):
    """The dense [N, K] weight in `dtype` from a bitsandbytes NF4 parameter."""
    packed = packed.view(-1)
    n_bytes = packed.numel()
    N, K = quant_state.shape
    out = torch.empty((N * K,), dtype=dtype, device=packed.device)
    code = quant_state.state2.code
    if code.dtype != torch.float32:
        code = code.float()
    scale2 = quant_state.state2.absmax
    if scale2.dtype != torch.float32:
        scale2 = scale2.float()
    offset = float(quant_state.offset)
    BYTES = 1024  # 2048 values = 32 blocks of 64 per program (measured: on par with bitsandbytes)
    grid = (triton.cdiv(n_bytes, BYTES),)
    nf4_dequant_kernel[grid](packed, quant_state.absmax, code, scale2, offset,
                             _nf4_table(packed.device), out, n_bytes, BLOCK=64, BLOCK2=256,
                             BYTES=BYTES, num_warps=8)
    return out.view(N, K)


class QuantLinear(torch.autograd.Function):
    """out = x @ W^T with W unpacked from NF4 on the fly, forward and backward; W is frozen."""

    @staticmethod
    def forward(ctx, x, packed, quant_state):
        w = dequantize_nf4(packed, quant_state, x.dtype)
        out = torch.matmul(x, w.t())
        ctx.packed, ctx.quant_state = packed, quant_state
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w = dequantize_nf4(ctx.packed, ctx.quant_state, grad_out.dtype)
        return torch.matmul(grad_out, w), None, None


def quant_linear(x, packed, quant_state):
    return QuantLinear.apply(x, packed, quant_state)
