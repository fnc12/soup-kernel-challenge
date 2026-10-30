"""The decoder on the real tokens only: padding removed inside the batch.

TRL pads every batch to its longest row; with these rows (mean 207 tokens, median 140, the
longest 512) 42% of the positions a step computes are padding. Removing them inside the batch is
allowed by the rules (rows are never mixed across batches). Here `Qwen2Model.forward` gathers the
non-pad positions once after the embedding, every layer (norms, the merged projections, the LoRA
branches, the MLP, the output head) works on the [1, N_real, D] rows, and only the attention
scatters q, k, v back into the padded [B, T] layout for SDPA (the same causal + padding mask as the
base run) and gathers its output rows again. Pad positions contribute nothing to the base run's
loss, metrics or gradients (their labels are -100 and the metrics are masked), so the numbers
stay the same; only the arithmetic on padding disappears.

The LoRA dropout masks are drawn on the padded [B, T, D] shape, as the base run draws them, and
gathered: the same random stream, the same masks.
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
        causal = modeling_qwen2.create_causal_mask(config=self.config, inputs_embeds=emb,
                                                   attention_mask=attention_mask,
                                                   past_key_values=None, position_ids=pos)
        cos, sin = self.rotary_emb(emb, pos)  # [1, T, head_dim]
        idx = attention_mask.reshape(-1).nonzero(as_tuple=True)[0]
        hidden = emb.reshape(B * T, D).index_select(0, idx).unsqueeze(0)
        tpos = idx % T
        packed_pe = (cos[0].index_select(0, tpos).unsqueeze(0),
                     sin[0].index_select(0, tpos).unsqueeze(0))
        pack = (idx, B, T)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden = decoder_layer(hidden, attention_mask=causal, position_embeddings=packed_pe,
                                   position_ids=pos, past_key_values=None, use_cache=False,
                                   pack=pack, **kwargs)
        hidden = self.norm(hidden)
        self._solution_pack = pack
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)

    modeling_qwen2.Qwen2Model.forward = forward


def scatter_rows(x, idx, rows):
    """[1, N_real, W] -> [rows, W] with zeros at the missing positions."""
    W = x.shape[-1]
    return x.new_zeros((rows, W)).index_copy(0, idx, x.reshape(-1, W))


def gather_rows(x, idx):
    """[rows, W] -> [1, N_real, W]."""
    return x.index_select(0, idx).unsqueeze(0)
