"""Rotary position embedding for q and k in one Triton kernel each way.

transformers applies RoPE as `q * cos + rotate_half(q) * sin` with `rotate_half` = slice, negate,
concatenate: about ten element-wise launches per layer for q and k together, every one of them
promoting the fp16 projections to fp32 temporaries that SDPA then casts down again, and the
backward pass runs the mirror image of that chain. Here one program rotates the heads of one
position: it reads the two halves of each head and the position's cos/sin once, computes in fp32,
and writes the rotated head in the dtype SDPA wants. The backward kernel applies the transpose
rotation to the incoming gradient. Nothing but cos and sin (which transformers keeps anyway) is
saved for the backward pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_heads, T, stride_x, stride_cb, stride_ct,
                stride_o, HALF: tl.constexpr, HEADS: tl.constexpr, BACKWARD: tl.constexpr):
    """One program: one position (row = b * T + t), HEADS heads of 2 * HALF values each. cos/sin
    are [Bc, T, 2 * HALF] with Bc = 1 (stride_cb = 0) or the batch size."""
    row = tl.program_id(0)
    crow = (row // T) * stride_cb + (row % T) * stride_ct
    hb = tl.program_id(1) * HEADS
    heads = hb + tl.arange(0, HEADS)[:, None]
    d = tl.arange(0, HALF)[None, :]
    inside = heads < n_heads
    base = row * stride_x + heads * (2 * HALF) + d
    x1 = tl.load(x_ptr + base, mask=inside, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF, mask=inside, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + crow + d).to(tl.float32)
    s = tl.load(sin_ptr + crow + d).to(tl.float32)
    if BACKWARD:
        # transpose of the rotation: (dy1 * c + dy2 * s, dy2 * c - dy1 * s)
        o1 = x1 * c + x2 * s
        o2 = x2 * c - x1 * s
    else:
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
    obase = row * stride_o + heads * (2 * HALF) + d
    tl.store(out_ptr + obase, o1.to(out_ptr.dtype.element_ty), mask=inside)
    tl.store(out_ptr + obase + HALF, o2.to(out_ptr.dtype.element_ty), mask=inside)


def _launch(x, cos, sin, head_dim, backward, out_dtype):
    """x: [B, T, H * head_dim] contiguous; cos, sin: [1 or B, T, head_dim] (cat(freqs, freqs))."""
    B, T, width = x.shape
    n_heads = width // head_dim
    x2 = x.reshape(B * T, width)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    stride_cb = 0 if cos.shape[0] == 1 else cos.stride(0)
    out = torch.empty((B * T, width), dtype=out_dtype, device=x.device)
    HEADS = min(8, triton.next_power_of_2(n_heads))
    grid = (B * T, triton.cdiv(n_heads, HEADS))
    rope_kernel[grid](x2, cos, sin, out, n_heads, T, x2.stride(0), stride_cb, cos.stride(1),
                      out.stride(0), HALF=head_dim // 2, HEADS=HEADS, BACKWARD=backward,
                      num_warps=2)
    return out.view(B, T, width)


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, head_dim, out_dtype):
        ctx.save_for_backward(cos, sin)
        ctx.head_dim, ctx.in_dtype = head_dim, x.dtype
        return _launch(x, cos, sin, head_dim, False, out_dtype)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _launch(dy, cos, sin, ctx.head_dim, True, ctx.in_dtype), None, None, None, None


def rotary(x, cos, sin, head_dim, out_dtype=None):
    """x: [B, T, H * head_dim] -> rotated, same layout, in `out_dtype` (default: x's)."""
    if cos.dim() == 4:  # a broadcast [B, 1, T, D] from apply_rotary_pos_emb's unsqueeze
        cos, sin = cos.squeeze(1), sin.squeeze(1)
    if cos.shape[0] not in (1, x.shape[0]) or cos.shape[1] != x.shape[1] or cos.shape[2] != head_dim:
        raise ValueError(f"cos/sin {tuple(cos.shape)} do not match x {tuple(x.shape)}")
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
    return Rotary.apply(x, cos, sin, head_dim, out_dtype or x.dtype)
