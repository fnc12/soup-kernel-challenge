"""The frozen embedding table in the compute dtype, the residual stream still in fp32.

`prepare_model_for_kbit_training` upcasts the (frozen, tied) embedding / `lm_head` table of
Qwen2.5-1.5B to fp32: 151,936 x 1,536 x 4 bytes = 0.93 GB that nothing reads in fp32. The output
head runs under autocast, i.e. from an fp16 copy (another 0.47 GB each step), and the embedding
lookup feeds an fp32 residual stream that we keep.

After Soup has built the model, this keeps that table in the compute dtype (fp16 on a T4: every
bf16 checkpoint value with |v| >= 2^-14 converts exactly), has the lookup return fp32 rows as
before, and lets the fused head (solution/cross_entropy.py) read the table directly instead of
making its own fp16 copy. The adapter, the loss and every logged number stay the same.
"""

import torch


def shrink_embedding(model, dtype):
    """Cast the tied input/output embedding of a built model to `dtype`; the lookup upcasts to fp32."""
    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    if not callable(get_in) or not callable(get_out):
        return False
    embed, head = get_in(), get_out()
    if not isinstance(embed, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        return False
    w = embed.weight
    if not isinstance(w, torch.Tensor) or not w.is_cuda or w.dtype != torch.float32 \
            or w.requires_grad or head.weight.requires_grad:
        return False
    tied = head.weight is w or head.weight.data_ptr() == w.data_ptr()
    w.data = w.data.to(dtype)
    if not tied:
        head.weight.data = head.weight.data.to(dtype)
    else:
        head.weight = w  # the same parameter again, now in `dtype`

    def upcast(module, inputs, output):
        return output.float() if isinstance(output, torch.Tensor) else output

    embed.register_forward_hook(upcast)
    return True


def patch_soup_setup(dtype):
    """Run `shrink_embedding` when Soup's `_setup_transformers` hands back the finished model."""
    try:
        from soup_cli.trainer import sft
    except Exception:  # noqa: BLE001
        return False
    cls = getattr(sft, "SFTTrainerWrapper", None)
    original = getattr(cls, "_setup_transformers", None)
    if cls is None or original is None:
        return False

    def _setup_transformers(self, cfg, tcfg):
        result = original(self, cfg, tcfg)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            try:
                shrink_embedding(model, dtype)
            except Exception as exc:  # noqa: BLE001 - never break the setup over a memory saving
                print(f"solution: embedding left in fp32 ({exc!r})")
            try:
                from .fused_mlp import register_layers

                register_layers(model)
            except Exception as exc:  # noqa: BLE001
                print(f"solution: layer indices unknown ({exc!r})")
        return result

    cls._setup_transformers = _setup_transformers
    return True
