"""Студия Евгения Захарова: Triton kernels for `soup train` on a Kaggle T4.

`apply()` runs once, before Soup builds the model and the trainer. It patches, narrowly:

- `Qwen2RMSNorm.forward`            -> fused RMSNorm (solution/rmsnorm.py), fp16 output;
- `Qwen2ForCausalLM.forward`        -> fused head + cross-entropy in row chunks, no logits tensor
                                       (solution/cross_entropy.py);
- `trl.SFTTrainer.compute_loss`     -> the same `entropy` / `mean_token_accuracy` / `num_tokens`
                                       columns from the fused kernel, and the per-row losses for
                                       `rewind.jsonl` through Soup's own `_soup_chunked_step` slot;
- `SFTTrainerWrapper._setup_transformers` -> the frozen embedding / head table kept in the compute
                                       dtype, the residual stream still fp32 (solution/embed.py);
- `Qwen2MLP.forward`                -> one merged gate|up NF4 matmul, a 2-D SwiGLU kernel over that
                                       buffer, the down projection; the last 14 layers rebuild their
                                       gate|up in the backward pass (solution/fused_mlp.py);
- `Qwen2Attention.forward`          -> one merged q|k|v NF4 matmul + PEFT's LoRA branches, RoPE in
                                       one kernel (solution/attention.py, solution/rope.py);
- `Qwen2Model.forward`              -> the decoder on the real tokens only, padding removed inside
                                       the batch (solution/packed.py);
- `bitsandbytes Linear4bit.forward` -> our NF4 unpacking kernel + cuBLAS, result left in fp16
                                       (solution/lowprec.py, solution/quant.py).

A background thread compiles the kernels while Soup loads the model (solution/warmup.py).
With `SOLUTION_STREAM=1` the 4-bit weights wait in pinned host memory between uses and reach the
GPU one layer ahead of their matmul (solution/stream.py).

Every patch checks that it is looking at a real CUDA tensor / module and otherwise falls through
to the original code, so Soup's regression tests (fake modules, CPU) run unchanged.
"""

import os

_APPLIED = False
DEFAULTS = {"SOLUTION_STREAM": "1", "SOLUTION_OFFLOAD": "1"}  # what runs without any environment variable


def setting(name):
    return os.environ.get(name, DEFAULTS.get(name, "0"))


def apply():
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return
    from .startup import mark

    mark("apply")
    if os.environ.get("SOLUTION_DIAG_MEM") == "1":
        try:
            import torch

            torch.cuda.memory._record_memory_history(max_entries=20000)
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("SOLUTION_NO_STARTUP") != "1":
        from .startup import cache_package_versions, datasets_caching_off

        try:
            if os.environ.get("SOLUTION_PACKAGE_CACHE") == "1":
                cache_package_versions()  # measured on a T4: the walk itself costs more than it saves
            datasets_caching_off()
        except Exception as exc:  # noqa: BLE001
            print(f"solution: start-up shortcuts off ({exc!r})")
    try:
        from .rmsnorm import patch_qwen2_rmsnorm
        from .cross_entropy import patch_qwen2_causal_lm, patch_sft_trainer
        from .lowprec import patch_bnb_linear4bit
        from .embed import patch_soup_setup
        from .fused_mlp import patch_qwen2_mlp
        from .attention import patch_qwen2_attention
        from .packed import patch_qwen2_model
    except Exception as exc:  # noqa: BLE001 - without triton or trl there is nothing to patch
        if os.environ.get("SOLUTION_STRICT"):
            raise
        print(f"solution: kernels not applied ({exc!r})")
        return
    from .lowprec import activation_dtype

    mark("imports")
    if os.environ.get("SOLUTION_NO_STARTUP") != "1":
        try:
            from .startup import patch_allocator_warmup, patch_kbit_upcast, patch_loader_cast

            patch_kbit_upcast()
            patch_loader_cast()
            patch_allocator_warmup()
        except Exception as exc:  # noqa: BLE001
            print(f"solution: k-bit upcast left on the GPU ({exc!r})")
    if os.environ.get("SOLUTION_NO_RMSNORM") != "1":
        patch_qwen2_rmsnorm(activation_dtype())
    if os.environ.get("SOLUTION_NO_CE") != "1":
        patch_qwen2_causal_lm(activation_dtype())
        patch_sft_trainer()
    if setting("SOLUTION_STREAM") == "1":
        try:
            import torch

            if torch.cuda.is_available():
                from . import quant
                from .stream import Streamer

                quant.STREAMER = Streamer(ahead=int(os.environ.get("SOLUTION_STREAM_AHEAD", "4")),
                                          ring_mb=int(os.environ.get("SOLUTION_STREAM_RING_MB", "64")))
                if os.environ.get("SOLUTION_STREAM_LOAD") != "0":
                    from .stream import patch_load

                    patch_load()  # the weights leave the GPU as the model loads
        except Exception as exc:  # noqa: BLE001
            print(f"solution: weight streaming off ({exc!r})")
    if setting("SOLUTION_OFFLOAD") == "1":
        try:
            from .activations import start as offload_start

            offload_start()  # saved activations wait in pinned host memory
        except Exception as exc:  # noqa: BLE001
            print(f"solution: activation offload off ({exc!r})")
    if os.environ.get("SOLUTION_NO_LOWPREC") != "1":
        patch_bnb_linear4bit(own_dequant=os.environ.get("SOLUTION_NO_NF4") != "1")
    if os.environ.get("SOLUTION_NO_EMBED") != "1":
        patch_soup_setup(activation_dtype())
    if os.environ.get("SOLUTION_NO_MLP") != "1":
        rc = os.environ.get("SOLUTION_RECOMPUTE", "14")  # the last 14 layers: measured as free
        patch_qwen2_mlp(activation_dtype(), recompute=(28 if rc == "1" else int(rc or 0)))
    if os.environ.get("SOLUTION_NO_WARMUP") != "1":
        from .warmup import start as warmup_start

        warmup_start(activation_dtype())  # compiles the kernels while Soup loads the model
    if os.environ.get("SOLUTION_NO_ATTN") != "1":
        patch_qwen2_attention(activation_dtype(), fused_rope=os.environ.get("SOLUTION_NO_ROPE") != "1")
        if os.environ.get("SOLUTION_NO_PACK") != "1" and os.environ.get("SOLUTION_NO_CE") != "1":
            patch_qwen2_model()  # padding removed inside the batch; needs the attention and head patches
    mark("applied")
