"""Solo AI Team: Triton kernels for `soup train` on a Kaggle T4.

`apply()` runs once, before Soup builds the model and the trainer. It patches, narrowly:

- `Qwen2RMSNorm.forward`            -> fused RMSNorm (solution/rmsnorm.py), fp16 output;
- `Qwen2ForCausalLM.forward`        -> fused head + cross-entropy in row chunks, no logits tensor
                                       (solution/cross_entropy.py);
- `trl.SFTTrainer.compute_loss`     -> the same `entropy` / `mean_token_accuracy` / `num_tokens`
                                       columns from the fused kernel, and the per-row losses for
                                       `rewind.jsonl` through Soup's own `_soup_chunked_step` slot;
- `bitsandbytes Linear4bit.forward` -> no cast of the fp16 result back to fp32 (solution/lowprec.py);
- `SFTTrainerWrapper._setup_transformers` -> the frozen embedding / head table kept in the compute
                                       dtype, the residual stream still fp32 (solution/embed.py);
- `Qwen2MLP.forward`                -> the kit's example SwiGLU (its kernels do not count as ours).

Every patch checks that it is looking at a real CUDA tensor / module and otherwise falls through
to the original code, so Soup's regression tests (fake modules, CPU) run unchanged.
"""

import os

_APPLIED = False


def apply():
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return
    from .swiglu import patch_qwen2_mlp

    patch_qwen2_mlp()  # the example: works out of the box, does not count as our own kernel
    try:
        from .rmsnorm import patch_qwen2_rmsnorm
        from .cross_entropy import patch_qwen2_causal_lm, patch_sft_trainer
        from .lowprec import patch_bnb_linear4bit
        from .embed import patch_soup_setup
    except Exception as exc:  # noqa: BLE001 - without triton or trl there is nothing to patch
        if os.environ.get("SOLUTION_STRICT"):
            raise
        print(f"solution: kernels not applied ({exc!r})")
        return
    from .lowprec import activation_dtype

    if os.environ.get("SOLUTION_NO_RMSNORM") != "1":
        patch_qwen2_rmsnorm(activation_dtype())
    if os.environ.get("SOLUTION_NO_CE") != "1":
        patch_qwen2_causal_lm(activation_dtype())
        patch_sft_trainer()
    if os.environ.get("SOLUTION_NO_LOWPREC") != "1":
        patch_bnb_linear4bit(own_dequant=os.environ.get("SOLUTION_NO_NF4") != "1")
    if os.environ.get("SOLUTION_NO_EMBED") != "1":
        patch_soup_setup(activation_dtype())
