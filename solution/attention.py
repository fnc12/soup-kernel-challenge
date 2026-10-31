"""Qwen2 attention with q, k and v as one merged 4-bit matmul.

Base Soup runs q_proj, k_proj and v_proj as three 4-bit linears on the same input: three
unpackings, three GEMMs (two of them only 256 columns wide, a poor shape for tensor cores), three
bias adds and three casts back to fp32, plus PEFT's LoRA branch on q and v. Here the three packed
weights are stacked once into one [2048, 1536] NF4 weight (solution/quant.py): one unpacking, one
GEMM, one bias add, then the LoRA branches exactly as PEFT computes them (same modules, same
dropout order, so the same random masks), then the rotary embedding in one kernel
(solution/rope.py), SDPA and o_proj as before.
"""

import torch


def _lora_add(module, x, part, pack=None):
    """PEFT's vanilla LoRA branch of `module` added to `part` (its base result), or `part` itself
    when the module is a plain linear. With `pack` (rows gathered from a padded [B, T] batch) the
    dropout mask is drawn on the padded shape, as the base run draws it, and gathered."""
    lora_A = getattr(module, "lora_A", None)
    if lora_A is None:
        return part
    for adapter in module.active_adapters:
        if adapter not in lora_A:
            continue
        a, b = lora_A[adapter], module.lora_B[adapter]
        dropout, scaling = module.lora_dropout[adapter], module.scaling[adapter]
        if isinstance(dropout, torch.nn.Dropout) and dropout.p > 0 and module.training:
            # the base run calls dropout on the fp32 [B, T, D] norm output; PyTorch's fused
            # dropout maps its random numbers to elements by dtype (4 per thread for fp32, 8 for
            # fp16), so the mask is drawn on fp32 ones of that shape and gathered: same stream,
            # same masks
            D = x.shape[-1]
            if pack is not None:
                idx, B, T = pack[:3]
                mask = dropout(torch.ones((B, T, D), dtype=torch.float32, device=x.device))
                xd = x * mask.view(B * T, D).index_select(0, idx).unsqueeze(0).to(x.dtype)
            else:
                mask = dropout(torch.ones(x.shape, dtype=torch.float32, device=x.device))
                xd = x * mask.to(x.dtype)
        else:
            xd = dropout(x)
        if getattr(b, "bias", None) is None and part.dim() == 3 and part.shape[0] == 1:
            # lora_B, the scaling and the add in one addmm (autocast runs it in the compute dtype)
            part = torch.addmm(part[0], a(xd)[0], b.weight.t(), alpha=scaling).unsqueeze(0)
        else:
            part = part + b(a(xd)) * scaling
    return part


def _vanilla(module):
    """True for a plain linear or a PEFT LoRA layer in its ordinary state (no variants, not
    merged, adapters on)."""
    if not hasattr(module, "lora_A"):
        return True
    if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return False
    variants = getattr(module, "lora_variant", {})
    return not any(adapter in variants for adapter in module.active_adapters)


_PADDED = __import__("os").environ.get("SOLUTION_ATTN", "padded") == "padded"


def merged_qkv(module, bases, dtype=torch.float16):
    """The cached merged q|k|v QuantWeight, bias and row sizes of one attention module."""
    from . import quant

    key = tuple(id(m.weight) for m in bases) + tuple(m.bias.data_ptr() for m in bases)
    cached = getattr(module, "_solution_qkv", None)
    if cached is None or getattr(module, "_solution_qkv_key", None) != key:
        qw = quant.QuantWeight([m.weight for m in bases], release=True)
        bias = torch.cat([m.bias.detach() for m in bases]).to(dtype)
        cached = (qw, bias, [m.weight.quant_state.shape[0] for m in bases])
        module._solution_qkv, module._solution_qkv_key = cached, key
    return cached


def patch_qwen2_attention(dtype=torch.float16, fused_rope=True):
    from transformers.models.qwen2 import modeling_qwen2

    from . import packed, quant, rope

    try:
        from bitsandbytes.nn import Linear4bit
    except Exception:  # noqa: BLE001
        Linear4bit = ()  # noqa: N806

    original = modeling_qwen2.Qwen2Attention.forward

    def merged(self, bases):
        return merged_qkv(self, bases, dtype)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        pack = kwargs.pop("pack", None)
        mods = [getattr(self, n, None) for n in ("q_proj", "k_proj", "v_proj")]
        bases = [getattr(m, "base_layer", m) for m in mods]
        if pack is not None and (past_key_values is not None or not isinstance(hidden_states, torch.Tensor)
                                 or not self.training or self.attention_dropout):
            raise RuntimeError("solution: packed rows reached an attention that cannot take them")
        if past_key_values is not None or kwargs.get("adapter_names") is not None \
                or not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda \
                or not all(isinstance(b, Linear4bit) and b.bias is not None for b in bases) \
                or not all(quant.supported(b.weight.quant_state) for b in bases) \
                or not all(_vanilla(m) for m in mods):
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        qw, bias, sizes = merged(self, bases)
        x16 = hidden_states if hidden_states.dtype == dtype else hidden_states.to(dtype)
        qkv = quant.quant_linear(x16, qw) + bias
        q, k, v = torch.split(qkv, sizes, dim=-1)
        q = _lora_add(mods[0], x16, q, pack)
        k = _lora_add(mods[1], x16, k, pack)
        v = _lora_add(mods[2], x16, v, pack)

        input_shape = hidden_states.shape[:-1]
        cos, sin = position_embeddings
        if pack is not None:
            # rotate the real rows (cos/sin are already gathered per row), then causal SDPA per
            # sequence on its own rows: the same attention as the base run's causal + padding
            # mask, without the pad blocks it computes and without the [B, T] zero buffers
            idx, B, T, lengths = pack
            q = rope.rotary(q, cos, sin, self.head_dim, dtype)
            k = rope.rotary(k, cos, sin, self.head_dim, dtype)
            dropout_p = self.attention_dropout if self.training else 0.0
            if _PADDED and not dropout_p and packed._right_padded(idx, B, T, lengths):
                attn_output = packed.padded_attention(q, k, v, pack, self.head_dim,
                                                      self.num_key_value_groups, self.scaling)
            else:
                attn_output = packed.per_row_attention(q, k, v, lengths, self.head_dim,
                                                       self.num_key_value_groups, self.scaling,
                                                       dropout_p)
            attn_output = self.o_proj(attn_output)
            return attn_output, None
        elif fused_rope and cos.dim() == 3 and cos.shape[0] in (1, input_shape[0]) \
                and cos.shape[1] == input_shape[1] and cos.shape[-1] == self.head_dim \
                and self.head_dim % 2 == 0:
            hidden_shape = (*input_shape, -1, self.head_dim)
            q = rope.rotary(q, cos, sin, self.head_dim, dtype)
            k = rope.rotary(k, cos, sin, self.head_dim, dtype)
            query_states = q.view(hidden_shape).transpose(1, 2)
            key_states = k.view(hidden_shape).transpose(1, 2)
        else:
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = q.view(hidden_shape).transpose(1, 2)
            key_states = k.view(hidden_shape).transpose(1, 2)
            query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states,
                                                                           cos, sin)
        value_states = v.view(hidden_shape).transpose(1, 2)
        attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_qwen2.Qwen2Attention.forward = forward
