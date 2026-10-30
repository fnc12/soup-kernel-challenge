"""The decoder on the real tokens only: padding removed inside the batch.

TRL pads every batch to its longest row; with these rows (mean 207 tokens, median 140, the
longest 512) 42% of the positions a step computes are padding. Removing them inside the batch is
allowed by the rules (rows are never mixed across batches). Here `Qwen2Model.forward` gathers the
non-pad positions once after the embedding, and every layer (norms, the merged projections, the
LoRA branches, the MLP, the output head) works on the [1, N_real, D] rows. The attention runs
causal SDPA per sequence on that sequence's rows: the same attention the base run computes through
its causal + padding mask, minus the pad blocks the masked kernel still visits. Pad positions
contribute nothing to the base run's loss, metrics or gradients (their labels are -100 and the
metrics are masked), so the numbers stay the same; only the arithmetic on padding disappears.

The LoRA dropout masks are drawn on fp32 ones of the padded [B, T, D] shape, as the base run draws
them on its fp32 norm output, and gathered: the same random stream, the same masks.
"""

import torch


def patch_qwen2_model():
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.modeling_outputs import BaseModelOutputWithPast

    original = modeling_qwen2.Qwen2Model.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        if not self.training or past_key_values is not None or use_cache or position_ids is not None \
                or inputs_embeds is not None or not isinstance(input_ids, torch.Tensor) \
                or not input_ids.is_cuda or not isinstance(attention_mask, torch.Tensor) \
                or attention_mask.dim() != 2 or attention_mask.shape != input_ids.shape \
                or getattr(self, "has_sliding_layers", False):
            self._solution_pack = None
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        emb = self.embed_tokens(input_ids)
        B, T, D = emb.shape
        pos = torch.arange(T, device=emb.device).unsqueeze(0)
        causal = None  # the packed attention is causal per row; no [B, 1, T, T] mask is built
        cos, sin = self.rotary_emb(emb, pos)  # [1, T, head_dim]
        idx = attention_mask.reshape(-1).nonzero(as_tuple=True)[0]
        hidden = emb.reshape(B * T, D).index_select(0, idx).unsqueeze(0)
        tpos = idx % T
        packed_pe = (cos[0].index_select(0, tpos).unsqueeze(0),
                     sin[0].index_select(0, tpos).unsqueeze(0))
        lengths = attention_mask.sum(dim=1).tolist()  # one host sync per step, before the layers
        pack = (idx, B, T, lengths)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden = decoder_layer(hidden, attention_mask=causal, position_embeddings=packed_pe,
                                   position_ids=pos, past_key_values=None, use_cache=False,
                                   pack=pack, **kwargs)
        hidden = self.norm(hidden)
        self._solution_pack = pack
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)

    modeling_qwen2.Qwen2Model.forward = forward


def per_row_attention(q, k, v, lengths, head_dim, groups, scaling, dropout_p):
    """Causal SDPA of each sequence on its own rows. q, k, v: [1, N_real, heads * head_dim] packed
    row-major by sequence; `lengths` the sequence lengths in order. Returns [1, N_real, Hq * D]."""
    outs, start = [], 0
    q2, k2, v2 = q[0], k[0], v[0]
    for L in lengths:
        if L == 0:
            continue
        stop = start + L
        qi = q2[start:stop].view(1, L, -1, head_dim).transpose(1, 2)
        ki = k2[start:stop].view(1, L, -1, head_dim).transpose(1, 2)
        vi = v2[start:stop].view(1, L, -1, head_dim).transpose(1, 2)
        if groups > 1:  # the same as transformers' repeat_kv for grouped-query attention
            ki = ki.repeat_interleave(groups, dim=1)
            vi = vi.repeat_interleave(groups, dim=1)
        oi = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, dropout_p=dropout_p,
                                                              is_causal=L > 1, scale=scaling)
        outs.append(oi.transpose(1, 2).reshape(L, -1))
        start = stop
    return torch.cat(outs, dim=0).unsqueeze(0)


def scatter_rows(x, idx, rows):
    """[1, N_real, W] -> [rows, W] with zeros at the missing positions."""
    W = x.shape[-1]
    return x.new_zeros((rows, W)).index_copy(0, idx, x.reshape(-1, W))


def gather_rows(x, idx):
    """[rows, W] -> [1, N_real, W]."""
    return x.index_select(0, idx).unsqueeze(0)
