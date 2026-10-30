"""Solo AI Team: Triton kernels for `soup train` on a Kaggle T4.

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
                                       buffer, the down projection (solution/fused_mlp.py);
- `Qwen2Attention.forward`          -> one merged q|k|v NF4 matmul + PEFT's LoRA branches, RoPE in
                                       one kernel (solution/attention.py, solution/rope.py);
- `Qwen2Model.forward`              -> the decoder on the real tokens only, padding removed inside
                                       the batch (solution/packed.py);
- `bitsandbytes Linear4bit.forward` -> our NF4 unpacking kernel + cuBLAS, result left in fp16
                                       (solution/lowprec.py, solution/quant.py).

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

    if os.environ.get("SOLUTION_NO_RMSNORM") != "1":
        patch_qwen2_rmsnorm(activation_dtype())
    if os.environ.get("SOLUTION_NO_CE") != "1":
        patch_qwen2_causal_lm(activation_dtype())
        patch_sft_trainer()
    if os.environ.get("SOLUTION_NO_LOWPREC") != "1":
        patch_bnb_linear4bit(own_dequant=os.environ.get("SOLUTION_NO_NF4") != "1")
    if os.environ.get("SOLUTION_NO_EMBED") != "1":
        patch_soup_setup(activation_dtype())
    if os.environ.get("SOLUTION_NO_MLP") != "1":
        patch_qwen2_mlp(activation_dtype(), recompute=os.environ.get("SOLUTION_RECOMPUTE") == "1")
    if os.environ.get("SOLUTION_NO_ATTN") != "1":
        patch_qwen2_attention(activation_dtype(), fused_rope=os.environ.get("SOLUTION_NO_ROPE") != "1")
        if os.environ.get("SOLUTION_NO_PACK") != "1" and os.environ.get("SOLUTION_NO_CE") != "1":
            patch_qwen2_model()  # padding removed inside the batch; needs the attention and head patches
