"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_example_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


def _rmsnorm_ref(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return w.float() * (xf * torch.rsqrt(var + eps))


@pytest.mark.parametrize("in_dtype,out_dtype", [(torch.float32, torch.float16),
                                                (torch.float32, torch.float32),
                                                (torch.float16, torch.float16)])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1000)])
def test_rmsnorm_matches_pytorch(in_dtype, out_dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=in_dtype, requires_grad=True)
    w = (1 + 0.1 * torch.randn(shape[-1], device="cuda")).requires_grad_(True)
    grad = torch.randn(shape, device="cuda", dtype=out_dtype)
    eps = 1e-6

    ref = _rmsnorm_ref(x, w, eps)
    ref.backward(grad.float())
    ref_dx, ref_dw = x.grad.float(), w.grad.float()
    x.grad = w.grad = None

    out = rmsnorm(x, w, eps, out_dtype)
    assert out.dtype == out_dtype
    out.backward(grad)
    tol = TOLERANCE[out_dtype]
    torch.testing.assert_close(out.float(), ref.to(out_dtype).float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(w.grad.float(), ref_dw, rtol=2e-2, atol=2e-2 * ref_dw.abs().max())


def test_rmsnorm_frozen_weight_has_no_weight_grad():
    from solution.rmsnorm import rmsnorm

    x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
    w = torch.ones(64, device="cuda")
    rmsnorm(x, w, 1e-6, torch.float16).float().sum().backward()
    assert x.grad is not None and w.grad is None


@pytest.mark.parametrize("shape,vocab", [((2, 64, 128), 1000), ((4, 512, 1536), 151936),
                                         ((1, 7, 96), 4097)])
def test_fused_head_cross_entropy_matches_pytorch(shape, vocab):
    from solution.cross_entropy import FusedHeadCrossEntropy, IGNORE_INDEX

    torch.manual_seed(0)
    B, T, D = shape
    hidden = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    weight = (torch.randn(vocab, D, device="cuda") * 0.02).to(torch.float16)
    labels = torch.randint(0, vocab, (B, T), device="cuda")
    labels[:, : T // 3] = IGNORE_INDEX  # prompt tokens
    labels[0, -1] = IGNORE_INDEX
    shift = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous()
    denominator = (shift != IGNORE_INDEX).sum().float()

    # the base run: fp16 matmul under autocast, fp32 loss, sum / denominator
    h = hidden.clone().requires_grad_(True)
    logits = torch.matmul(h.to(torch.float16), weight.t())
    ref_loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, vocab), shift.reshape(-1),
                                                 ignore_index=IGNORE_INDEX, reduction="sum") \
        / denominator
    (ref_loss * 3.0).backward()
    ref_grad = h.grad.clone()
    with torch.no_grad():
        logp = torch.log_softmax(logits.float(), dim=-1)
        ref_entropy = -(logp.exp() * logp).sum(-1)
        ref_hits = ((logits.argmax(-1) == shift) & (shift != IGNORE_INDEX)).sum()
        ref_token_loss = torch.where(shift != IGNORE_INDEX,
                                     -logp.gather(-1, shift.clamp(min=0).unsqueeze(-1)).squeeze(-1),
                                     torch.zeros_like(ref_entropy))

    h2 = hidden.clone().requires_grad_(True)
    steps = []
    loss = FusedHeadCrossEntropy.apply(h2, weight, shift, denominator, steps)
    (loss * 3.0).backward()
    step = steps[0]
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(h2.grad, ref_grad, rtol=2e-2, atol=2e-2 * ref_grad.abs().max())
    torch.testing.assert_close(step.token_entropy, ref_entropy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(step.token_loss, ref_token_loss, rtol=1e-4, atol=1e-4)
    assert int(step.correct) == int(ref_hits)
    assert int(step.labelled) == int((shift != IGNORE_INDEX).sum())


@pytest.mark.parametrize("shape", [(8960, 1536), (1536, 8960), (256, 1536), (152, 1536)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nf4_dequant_matches_bitsandbytes(shape, dtype):
    bnb = pytest.importorskip("bitsandbytes")
    from solution.nf4 import dequantize_nf4, quant_linear, supported

    torch.manual_seed(0)
    w = torch.randn(shape, device="cuda") * 0.05
    packed, state = bnb.functional.quantize_4bit(w, blocksize=64, compress_statistics=True,
                                                 quant_type="nf4")
    assert supported(state)
    ref = bnb.functional.dequantize_4bit(packed, state).to(dtype)
    out = dequantize_nf4(packed, state, dtype)
    assert out.shape == tuple(shape) and out.dtype == dtype
    # bitsandbytes rounds the second-level absmax once more than our fused fma does: a few
    # values per million differ by one unit in the last place of fp16
    ulp = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(out, ref, rtol=ulp, atol=ulp * 0.05)
    assert (out != ref).float().mean().item() < 1e-3

    x = torch.randn(3, 10, shape[1], device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(3, 10, shape[0], device="cuda", dtype=dtype)
    ref_out = torch.nn.functional.linear(x, ref)
    ref_out.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None
    out = quant_linear(x, packed, state)
    out.backward(grad)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-2, atol=1e-2)
