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

import contextlib
import os

import torch


def patch_qwen2_model():
    from transformers.models.qwen2 import modeling_qwen2

    from . import activations
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
        calls = getattr(self, "_solution_calls", 0)
        if calls < 2:
            from .startup import mark

            mark(f"forward-{calls}")
            self._solution_calls = calls + 1
            if calls == 0 and os.environ.get("SOLUTION_DIAG_MEM") == "1":
                _peak_watch()
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
        off = activations.ACTIVE if torch.is_grad_enabled() else None
        if off is not None:
            off.new_step()
            hooks = torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack)
        else:
            hooks = contextlib.nullcontext()
        with hooks:
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


_PEAK = {"mib": 0, "blocks": None, "thread": None}


def _peak_watch():
    """Diagnostics: a thread that keeps the block list of the highest allocation seen."""
    import threading
    import time

    def run():
        while True:
            try:
                mib = torch.cuda.memory_allocated() >> 20
                if mib > _PEAK["mib"] + 40:
                    _PEAK["mib"] = mib
                    _PEAK["blocks"] = _largest_blocks(collect=True)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.0005)

    if _PEAK["thread"] is None:
        _PEAK["thread"] = threading.Thread(target=run, daemon=True)
        _PEAK["thread"].start()
        import atexit
        import sys

        def report():
            print(f"[solution-diag] highest sampled allocation {_PEAK['mib']} MiB", file=sys.stderr)
            for line in _PEAK["blocks"] or []:
                print(f"[solution-diag] {line}", file=sys.stderr)

        atexit.register(report)


def _largest_blocks(collect=False):
    import sys

    try:
        snap = torch.cuda.memory_snapshot()
    except Exception as exc:  # noqa: BLE001
        print(f"[solution-diag] no snapshot ({exc!r})", file=sys.stderr)
        return []
    blocks = []
    for seg in snap:
        for b in seg.get("blocks", []):
            if b.get("state") == "active_allocated":
                frames = [f for f in (b.get("frames") or []) if str(f.get("filename", "")).endswith(".py")]
                where = "; ".join(f"{f.get('filename','?').split('/')[-1]}:{f.get('line','?')} {f.get('name','?')}"
                                  for f in frames[:5])
                blocks.append((b.get("size", 0), seg.get("stream", 0), where))
    blocks.sort(reverse=True)
    by_where = {}
    for size, stream, where in blocks:
        key = where.split("; ")[0] if where else "?"
        by_where[key] = by_where.get(key, 0) + size
    agg = sorted(by_where.items(), key=lambda kv: -kv[1])[:14]
    lines = [f"{size >> 20} MiB: {where}" for size, stream, where in blocks[:10]]
    lines += [f"SUM {size >> 20} MiB ({len(blocks)} blocks total) <- {key}" for key, size in agg]
    if collect:
        return lines
    for line in lines:
        print(f"[solution-diag] {line}", file=sys.stderr)
    return lines


_RIGHT = {}


def _right_padded(idx, B, T, lengths):
    """True when every sequence's real tokens are a prefix of its row (right padding): then a
    causal kernel alone keeps each real query on its own sequence's earlier keys."""
    key = (id(lengths), B, T)
    right = _RIGHT.get(key)
    if right is None:
        starts, s = [], 0
        for L in lengths:
            starts.append(s)
            s += L
        first = idx[torch.tensor([st for st, L in zip(starts, lengths) if L > 0], device=idx.device)]
        rows = torch.tensor([b for b, L in enumerate(lengths) if L > 0], device=idx.device)
        right = bool((first == rows * T).all().item())
        _RIGHT.clear()
        _RIGHT[key] = right
    return right


def padded_attention(q, k, v, pack, head_dim, groups, scaling):
    """Causal SDPA of the whole batch in one call: the packed rows are scattered to [B, T] (pads
    zero at the end of each row), the causal kernel keeps each real query on its own sequence's
    earlier keys (a padded query row sees pads too, its output is dropped), the outputs come back
    packed. The same attention as `per_row_attention` for every real row, with one kernel per
    layer instead of one per sequence, and four saved tensors per layer that fit the offload
    threshold. Needs right padding; `per_row_attention` serves the other case."""
    idx, B, T, lengths = pack
    rows = B * T
    qp = scatter_rows(q[0], idx, rows).view(B, T, -1, head_dim).transpose(1, 2)
    kp = scatter_rows(k[0], idx, rows).view(B, T, -1, head_dim).transpose(1, 2)
    vp = scatter_rows(v[0], idx, rows).view(B, T, -1, head_dim).transpose(1, 2)
    if groups > 1:
        kp = kp.repeat_interleave(groups, dim=1)
        vp = vp.repeat_interleave(groups, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(qp, kp, vp, is_causal=True, scale=scaling)
    out = out.transpose(1, 2).reshape(rows, -1)
    return gather_rows(out, idx).reshape(1, idx.numel(), -1)
