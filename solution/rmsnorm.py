"""Fused RMSNorm for Qwen2, forward and backward in Triton.

Base Soup runs `Qwen2RMSNorm.forward` as five PyTorch ops in fp32 (cast, square, mean, rsqrt,
scale) and keeps three fp32 copies of the activation for the backward pass. Here one kernel reads
the residual stream once, writes the normalized rows in the dtype the next matmul wants (fp16),
and the backward kernel recomputes everything from the saved input and one scalar per row, so
nothing but the input (which the residual stream keeps alive anyway) is stored.

The weights of the norms are frozen in this task (LoRA on q_proj/v_proj only), so the backward
skips the weight gradient unless autograd asks for it.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["stride_x", "stride_y", "D"])
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, stride_x, stride_y, D, eps,
                       BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = w * (x * rstd)
    tl.store(y_ptr + row * stride_y + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["stride_dy", "stride_x", "stride_dx", "D"])
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, stride_dy, stride_x,
                       stride_dx, D, WITH_DW: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    xn = x * rstd
    wdy = w * dy
    # d/dx of w * x * rstd(x): rstd * (wdy - xn * mean(wdy * xn))
    c = tl.sum(wdy * xn, axis=0) / D
    dx = rstd * (wdy - xn * c)
    tl.store(dx_ptr + row * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    if WITH_DW:
        tl.atomic_add(dw_ptr + cols, dy * xn, mask=mask)


def _block(D):
    return max(128, triton.next_power_of_2(D))


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype):
        shape = x.shape
        D = shape[-1]
        x2 = x.reshape(-1, D)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        rows = x2.shape[0]
        y = torch.empty((rows, D), dtype=out_dtype, device=x.device)
        rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
        BLOCK_D = _block(D)
        rmsnorm_fwd_kernel[(rows,)](x2, weight, y, rstd, x2.stride(0), y.stride(0), D, eps,
                                    BLOCK_D=BLOCK_D, num_warps=4 if BLOCK_D <= 2048 else 8)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        D = x2.shape[1]
        rows = x2.shape[0]
        dy2 = dy.reshape(rows, D)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        with_dw = bool(ctx.needs_input_grad[1])
        dw = torch.zeros((D,), dtype=torch.float32, device=x2.device) if with_dw else dx
        BLOCK_D = _block(D)
        rmsnorm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, dw, dy2.stride(0), x2.stride(0),
                                    dx.stride(0), D, WITH_DW=with_dw, BLOCK_D=BLOCK_D,
                                    num_warps=4 if BLOCK_D <= 2048 else 8)
        return dx.view(ctx.shape), (dw.to(weight.dtype) if with_dw else None), None, None


def rmsnorm(x, weight, eps, out_dtype=None):
    return RMSNormFunction.apply(x, weight, eps, out_dtype or x.dtype)


def patch_qwen2_rmsnorm(out_dtype=torch.float16):
    """Qwen2RMSNorm.forward through the fused kernel. On CUDA the normalized rows come out in
    `out_dtype` (what the 4-bit matmuls and the LoRA branch read anyway); elsewhere the original
    code runs."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda \
                or not isinstance(weight, torch.Tensor) or hidden_states.shape[-1] != weight.shape[0]:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, self.variance_epsilon, out_dtype)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
