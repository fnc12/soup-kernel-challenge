"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


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


@pytest.mark.parametrize("shape", [(8960, 1536), (1536, 8960), (256, 1536), (512, 1536)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nf4_dequant_matches_bitsandbytes(shape, dtype):
    bnb = pytest.importorskip("bitsandbytes")
    from solution.quant import QuantWeight, quant_linear, supported

    torch.manual_seed(0)
    w = torch.randn(shape, device="cuda") * 0.05
    packed, state = bnb.functional.quantize_4bit(w, blocksize=64, compress_statistics=True,
                                                 quant_type="nf4")
    assert supported(state)
    packed.quant_state = state
    qw = QuantWeight([packed])
    ref = bnb.functional.dequantize_4bit(packed, state).to(dtype)
    out = qw.dequantize(dtype)
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
    out = quant_linear(x, qw)
    out.backward(grad)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-2, atol=1e-2)


def test_merged_quant_weight_stacks_weights_with_their_own_offsets():
    bnb = pytest.importorskip("bitsandbytes")
    from solution.quant import QuantWeight

    torch.manual_seed(1)
    parts, refs = [], []
    for n, scale in ((1536, 0.05), (256, 0.5), (256, 0.01)):
        w = torch.randn(n, 1536, device="cuda") * scale
        packed, state = bnb.functional.quantize_4bit(w, blocksize=64, compress_statistics=True,
                                                     quant_type="nf4")
        packed.quant_state = state
        parts.append(packed)
        refs.append(bnb.functional.dequantize_4bit(packed, state).to(torch.float16))
    qw = QuantWeight(parts)
    out = qw.dequantize(torch.float16)
    ref = torch.cat(refs)
    assert out.shape == ref.shape == (2048, 1536)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=5e-5)
    assert (out != ref).float().mean().item() < 1e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1000)])
def test_gated_swiglu_matches_pytorch(dtype, shape):
    from solution.fused_mlp import SwiGLUGated

    torch.manual_seed(0)
    I = shape[-1]
    gu = torch.randn(*shape[:-1], 2 * I, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    gate, up = gu.float().split(I, dim=-1)
    ref = torch.nn.functional.silu(gate) * up
    ref.backward(grad.float())
    ref_dgu = gu.grad.clone().float()
    gu.grad = None
    out = SwiGLUGated.apply(gu)
    out.backward(grad)
    tol = 2e-3 if dtype == torch.float16 else 1.6e-2
    torch.testing.assert_close(out.float(), ref.to(dtype).float(), rtol=tol, atol=tol)
    torch.testing.assert_close(gu.grad.float(), ref_dgu.to(dtype).float(), rtol=tol, atol=tol)


def test_mlp_recompute_matches_the_kept_path():
    bnb = pytest.importorskip("bitsandbytes")
    from solution.fused_mlp import MLPRecompute, SwiGLUGated
    from solution.quant import QuantWeight, quant_linear

    torch.manual_seed(2)
    K, I = 1536, 8960
    qws = []
    for shape in ((I, K), (I, K), (K, I)):
        packed, state = bnb.functional.quantize_4bit(torch.randn(shape, device="cuda") * 0.02,
                                                     blocksize=64, compress_statistics=True,
                                                     quant_type="nf4")
        packed.quant_state = state
        qws.append(packed)
    qw_gu, qw_d = QuantWeight(qws[:2]), QuantWeight(qws[2:])
    x = torch.randn(2, 64, K, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2, 64, K, device="cuda", dtype=torch.float16)
    ref = quant_linear(SwiGLUGated.apply(quant_linear(x, qw_gu)), qw_d)
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None
    out = MLPRecompute.apply(x, qw_gu, qw_d)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_dtype,out_dtype", [(torch.float16, torch.float16),
                                                (torch.float32, torch.float16)])
@pytest.mark.parametrize("heads", [12, 2])
@pytest.mark.parametrize("cos_batch", [1, 2])
def test_rotary_matches_transformers(in_dtype, out_dtype, heads, cos_batch):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    from solution.rope import rotary

    torch.manual_seed(0)
    B, T, D = 2, 37, 128
    x = torch.randn(B, T, heads * D, device="cuda", dtype=in_dtype, requires_grad=True)
    pos = torch.arange(T, device="cuda").float()
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    freqs = torch.cat([pos[:, None] * inv[None, :]] * 2, dim=-1)
    cos = freqs.cos().expand(cos_batch, T, D).contiguous()
    sin = freqs.sin().expand(cos_batch, T, D).contiguous()
    grad = torch.randn(B, T, heads * D, device="cuda", dtype=out_dtype)

    q = x.view(B, T, heads, D).transpose(1, 2)
    ref, _ = apply_rotary_pos_emb(q, q, cos, sin)
    ref = ref.transpose(1, 2).reshape(B, T, heads * D)
    ref.backward(grad.to(ref.dtype))
    ref_dx = x.grad.clone()
    x.grad = None

    out = rotary(x, cos, sin, D, out_dtype)
    assert out.dtype == out_dtype
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=2e-3, atol=2e-3)


def test_per_row_attention_matches_masked_sdpa():
    from solution.packed import per_row_attention

    torch.manual_seed(3)
    B, T, Hq, Hkv, D = 3, 40, 12, 2, 128
    lengths = [40, 17, 1]
    mask = torch.zeros(B, T, dtype=torch.bool, device="cuda")
    for i, L in enumerate(lengths):
        mask[i, :L] = True
    q = torch.randn(B, T, Hq * D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, T, Hkv * D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, T, Hkv * D, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(B, T, Hq * D, device="cuda", dtype=torch.float16)
    # the base: causal + padding mask on the padded layout, keys/values repeated for GQA
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    full = causal[None, None] & mask[:, None, None, :]
    full = full | ~mask[:, None, :, None]  # pad queries attend to themselves to avoid NaN rows
    qb = q.view(B, T, Hq, D).transpose(1, 2)
    kb = k.view(B, T, Hkv, D).transpose(1, 2).repeat_interleave(Hq // Hkv, dim=1)
    vb = v.view(B, T, Hkv, D).transpose(1, 2).repeat_interleave(Hq // Hkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(qb, kb, vb, attn_mask=full, scale=D ** -0.5)
    ref = ref.transpose(1, 2).reshape(B, T, Hq * D)
    (ref * grad)[mask].sum().backward()
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = None
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    gather = lambda t: t.reshape(B * T, -1).index_select(0, idx).unsqueeze(0)
    out = per_row_attention(gather(q), gather(k), gather(v), lengths, D, Hq // Hkv, D ** -0.5, 0.0)
    (out[0] * gather(grad)[0]).sum().backward()
    torch.testing.assert_close(out[0], ref[mask], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(q.grad[mask], ref_dq[mask], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k.grad[mask], ref_dk[mask], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v.grad[mask], ref_dv[mask], rtol=2e-2, atol=2e-2)


def test_warmup_compiles_every_kernel():
    from solution.lowprec import activation_dtype
    from solution.warmup import _warm

    _warm(activation_dtype())  # raises if any kernel fails to compile or launch
