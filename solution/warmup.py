"""Compile the Triton kernels while Soup is still loading the model.

Every run starts with an empty Triton cache (the harness makes sure of it), and on Kaggle's two
CPU cores the first training step pays several seconds for the compilation of our seven kernels.
`apply()` runs before Soup loads the model, which takes tens of seconds of mostly CPU and disk
work, so a background thread launches each kernel once on tiny tensors with exactly the
compile-time configuration training uses (dtypes, constexprs, the divisibility class of the
integer arguments). The cache keys match, so the first real launch finds the compiled kernel.
Nothing about the training changes; if the thread fails for any reason, the kernels compile on
first use as before.
"""

import threading

import torch


def _warm(dtype):
    from . import cross_entropy, fused_mlp, quant, rmsnorm, rope

    dev = torch.device("cuda")
    # RMSNorm: D = 1536 (BLOCK 2048), fp32 input, `dtype` output, no weight gradient
    x = torch.randn(16, 1536, device=dev, dtype=torch.float32, requires_grad=True)
    w = torch.ones(1536, device=dev)
    y = rmsnorm.rmsnorm(x, w, 1e-6, dtype)
    y.backward(torch.ones_like(y))
    # gated SwiGLU on a merged [R, 2I] buffer, I = 8960
    gu = torch.randn(16, 2 * 8960, device=dev, dtype=dtype, requires_grad=True)
    h = fused_mlp.SwiGLUGated.apply(gu)
    h.backward(torch.ones_like(h))
    # rotary: head_dim 128, HEADS 4, forward and backward
    q = torch.randn(1, 16, 12 * 128, device=dev, dtype=dtype, requires_grad=True)
    cos = torch.randn(1, 16, 128, device=dev)
    r = rope.rotary(q, cos, cos, 128, dtype)
    r.backward(torch.ones_like(r))
    # the head + cross-entropy chunk kernel: V = 151936 columns, labels int32
    V = 151936
    logits = torch.randn(2, V, device=dev, dtype=dtype)
    grad = torch.empty_like(logits)
    labels = torch.tensor([5, cross_entropy.IGNORE_INDEX], device=dev, dtype=torch.int32)
    loss = torch.empty(2, device=dev)
    ent = torch.empty(2, device=dev)
    hits = torch.empty(2, device=dev, dtype=torch.int32)
    cross_entropy.head_ce_kernel[(2,)](logits, labels, grad, loss, ent, hits, logits.stride(0),
                                       grad.stride(0), V, IGNORE=cross_entropy.IGNORE_INDEX,
                                       BLOCK_V=4096, num_warps=8)
    # NF4 unpacking: a packed length that is a multiple of 16 bytes, like every weight here
    n_bytes = 64 * 256 // 2 * 16
    packed = torch.zeros(n_bytes, device=dev, dtype=torch.uint8)
    absmax = torch.zeros(n_bytes * 2 // 64, device=dev, dtype=torch.uint8)
    code = torch.zeros(256, device=dev)
    scale2 = torch.zeros(n_bytes * 2 // (64 * 256), device=dev)
    offsets = torch.zeros_like(scale2)
    out = torch.empty(n_bytes * 2, device=dev, dtype=dtype)
    quant.nf4_dequant_kernel[(n_bytes // 1024,)](packed, absmax, code, scale2, offsets,
                                                 quant._nf4_table(dev), out, n_bytes, BLOCK=64,
                                                 BLOCK2=256, BYTES=1024, num_warps=8)
    torch.cuda.synchronize()


def start(dtype):
    """Compile in the background; returns the thread (joined by nobody: it just finishes)."""
    if not torch.cuda.is_available():
        return None

    def run():
        try:
            _warm(dtype)
        except Exception as exc:  # noqa: BLE001 - compilation then happens on first use
            print(f"solution: kernel warm-up skipped ({exc!r})")

    t = threading.Thread(target=run, name="solution-warmup", daemon=True)
    t.start()
    return t
