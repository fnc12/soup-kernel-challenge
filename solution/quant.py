"""NF4 weights unpacked by one Triton kernel; several 4-bit weights merged into one.

bitsandbytes stores a 4-bit weight as packed nibbles (two per byte), one absmax per block of 64
values, and (double quantization) those absmax values themselves as uint8 codes with one fp32
scale per 256 of them plus an offset. For a batch this large (4 x 512 rows) bitsandbytes does not
run its fused GEMM: every projection first launches a kernel to rebuild the absmax values, then a
kernel to unpack the weight into a dense fp16 matrix, then cuBLAS; the backward pass unpacks the
same weight again for the gradient of the input.

`nf4_dequant_kernel` does both levels in one pass and writes the dense rows in the compute dtype.
`QuantWeight` is the view of one or several bitsandbytes 4-bit parameters as one [N, K] weight:
the packed bytes, absmax codes and scales of q/k/v (or gate/up) are concatenated once, and the
per-weight offsets become a per-scale-block array, so the merged weight unpacks in one launch and
multiplies in one GEMM. `QuantLinear` is the autograd function around it (the weights are frozen,
so only the input needs a gradient).
"""

import torch
import triton
import triton.language as tl

NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
       0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)
BLOCK = 64     # values per absmax
BLOCK2 = 256   # absmax codes per second-level scale


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_u8_ptr, code_ptr, scale2_ptr, offset_ptr, nf4_ptr,
                       out_ptr, n_bytes, BLOCK: tl.constexpr, BLOCK2: tl.constexpr,
                       BYTES: tl.constexpr):
    """BYTES packed bytes per program (2 * BYTES values, a whole number of BLOCK-value blocks)."""
    pid = tl.program_id(0)
    offs = pid * BYTES + tl.arange(0, BYTES)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    hi = byte >> 4
    lo = byte & 15
    # value index i = 2 * byte index (hi) and 2 * byte index + 1 (lo); its block is i // BLOCK
    block = (offs * 2) // BLOCK
    block2 = block // BLOCK2
    code = tl.load(absmax_u8_ptr + block, mask=mask, other=0).to(tl.int32)
    scale2 = tl.load(scale2_ptr + block2, mask=mask, other=0.0)
    offset = tl.load(offset_ptr + block2, mask=mask, other=0.0)
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
    """The layout this kernel handles: NF4, blocks of 64, double-quantized absmax in blocks of 256,
    and a whole number of scale blocks (so weights can be concatenated)."""
    if quant_state is None or getattr(quant_state, "quant_type", None) != "nf4" \
            or not getattr(quant_state, "nested", False) or quant_state.blocksize != BLOCK \
            or getattr(quant_state.state2, "blocksize", None) != BLOCK2 \
            or quant_state.absmax.dtype != torch.uint8 or quant_state.state2.code.numel() != 256:
        return False
    n, k = quant_state.shape
    return (n * k) % (BLOCK * BLOCK2) == 0


class QuantWeight:
    """One dense [N, K] weight made of one or more bitsandbytes NF4 parameters stacked along N."""

    def __init__(self, params):
        states = [p.quant_state for p in params]
        if not all(supported(s) for s in states):
            raise ValueError("unsupported quantization layout")
        k = states[0].shape[1]
        if any(s.shape[1] != k for s in states):
            raise ValueError("inner dimensions differ")
        code = states[0].state2.code.float()
        if any(not torch.equal(s.state2.code.float(), code) for s in states[1:]):
            raise ValueError("different absmax code tables")
        self.sizes = [s.shape[0] for s in states]
        self.N, self.K = sum(self.sizes), k
        self.device = params[0].device
        self.code = code
        self.nf4 = _nf4_table(self.device)
        if len(params) == 1:
            self.packed = params[0].data.view(-1)
            self.absmax = states[0].absmax.view(-1)
            self.scale2 = states[0].state2.absmax.float().view(-1)
        else:
            self.packed = torch.cat([p.data.view(-1) for p in params])
            self.absmax = torch.cat([s.absmax.view(-1) for s in states])
            self.scale2 = torch.cat([s.state2.absmax.float().view(-1) for s in states])
        self.offsets = torch.cat([torch.full((s.state2.absmax.numel(),), float(s.offset),
                                             dtype=torch.float32, device=self.device) for s in states])
        assert self.packed.numel() * 2 == self.N * self.K
        assert self.scale2.numel() == self.offsets.numel() == self.N * self.K // (BLOCK * BLOCK2)

    def dequantize(self, dtype):
        n_bytes = self.packed.numel()
        out = torch.empty((self.N * self.K,), dtype=dtype, device=self.device)
        BYTES = 1024  # 2048 values = 32 blocks per program (measured: on par with bitsandbytes)
        nf4_dequant_kernel[(triton.cdiv(n_bytes, BYTES),)](
            self.packed, self.absmax, self.code, self.scale2, self.offsets, self.nf4, out, n_bytes,
            BLOCK=BLOCK, BLOCK2=BLOCK2, BYTES=BYTES, num_warps=8)
        return out.view(self.N, self.K)


class QuantLinear(torch.autograd.Function):
    """out = x @ W^T with W unpacked from NF4 on the fly, forward and backward; W is frozen."""

    @staticmethod
    def forward(ctx, x, qw):
        w = qw.dequantize(x.dtype)
        out = torch.matmul(x, w.t())
        ctx.qw = qw
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w = ctx.qw.dequantize(grad_out.dtype)
        return torch.matmul(grad_out, w), None


def quant_linear(x, qw):
    return QuantLinear.apply(x, qw)


def quant_weight_of(module):
    """The cached QuantWeight of a bitsandbytes Linear4bit module (built on first use)."""
    qw = getattr(module, "_solution_qw", None)
    key = (module.weight.data_ptr(), tuple(module.weight.quant_state.shape))
    if qw is None or getattr(module, "_solution_qw_key", None) != key:
        qw = QuantWeight([module.weight])
        module._solution_qw = qw
        module._solution_qw_key = key
    return qw
