"""Fused output head + cross-entropy over the 151,936-token vocabulary, in row chunks.

Base Soup materializes the logits of the whole batch (4 x 512 x 151,936: 0.6 GB in fp16), upcasts
them to fp32 for the loss (1.2 GB), keeps the log-softmax for the backward pass (1.2 GB), then
TRL reads the same logits twice more for its `entropy` and `mean_token_accuracy` metrics and
Soup once more for the per-row losses of `rewind.jsonl`. Several gigabytes live at the same time,
and about 7% of the step goes to softmax-like passes over the vocabulary.

Here the head runs on `ROWS` positions at a time (the matmul in fp16, as autocast does in the base
run) and one Triton kernel does everything the base does with that chunk of logits in a single
read-and-write: log-sum-exp, the loss at the label, the gradient of the logits, the entropy of
every position and the argmax hit for the accuracy. The gradient on the hidden states is taken
chunk by chunk right away, so no logits outlive their chunk. The metrics are handed to the trainer
through the same slot Soup's own chunked-loss path uses (`_soup_chunked_step`), so TRL's columns
and `rewind.jsonl` get the same values, computed the same way.

The idea is the fused linear cross-entropy of Liger Kernel and Unsloth; the code is our own.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
ROWS = 256  # positions per chunk: logits chunk = ROWS x V x 2 bytes (78 MB at V = 151,936)


@triton.jit
def head_ce_kernel(logits_ptr, labels_ptr, grad_ptr, loss_ptr, entropy_ptr, hit_ptr,
                   stride_l, stride_g, V, IGNORE: tl.constexpr, BLOCK_V: tl.constexpr):
    """One program per row of the logits chunk: pass 1 finds the max and the sum of exponentials
    (online), pass 2 writes softmax - onehot into the gradient and sums p * logit for the
    entropy. Rows without a label (-100) still get their entropy; their gradient is zero."""
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    base = logits_ptr + row * stride_l
    m = -float("inf")
    s = 0.0
    best = -float("inf")
    best_idx = 0
    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        l = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        bm = tl.max(l, axis=0)
        m_new = tl.maximum(m, bm)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(l - m_new), axis=0)
        m = m_new
        # the first index of the block maximum, kept only if it beats the running best
        is_max = l == bm
        cand = tl.min(tl.where(is_max, cols, V), axis=0)
        best_idx = tl.where(bm > best, cand, best_idx)
        best = tl.maximum(best, bm)
    lse = m + tl.log(s)
    labelled = label != IGNORE
    safe_label = tl.where(labelled, label, 0)
    at_label = tl.load(base + safe_label).to(tl.float32)
    loss = tl.where(labelled, lse - at_label, 0.0)
    tl.store(loss_ptr + row, loss)
    tl.store(hit_ptr + row, (labelled & (best_idx == label)).to(tl.int32))
    pl = 0.0
    gbase = grad_ptr + row * stride_g
    for start in range(0, V, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < V
        l = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(l - lse)
        pl += tl.sum(tl.where(mask, p * l, 0.0), axis=0)
        g = tl.where(labelled, p - (cols == safe_label).to(tl.float32), 0.0)
        tl.store(gbase + cols, g.to(grad_ptr.dtype.element_ty), mask=mask)
    tl.store(entropy_ptr + row, lse - pl)


@dataclass
class HeadStep:
    """What the trainer's metrics read after a chunked step (the layout Soup's own chunked path
    leaves in `_soup_chunked_step`: `token_loss` is (B, T) with position t scoring labels[t+1])."""

    token_entropy: torch.Tensor
    token_loss: torch.Tensor
    correct: torch.Tensor
    labelled: torch.Tensor


class FusedHeadCrossEntropy(torch.autograd.Function):
    """loss = sum_t CE(hidden_t @ W^T, shift_labels_t) / denominator. The gradient on the hidden
    states is computed in `forward`, chunk by chunk, and only scaled in `backward`."""

    @staticmethod
    def forward(ctx, hidden, weight16, shift_labels, denominator, step_out):
        B, T, D = hidden.shape
        V = weight16.shape[0]
        n = B * T
        h = hidden.reshape(n, D)
        labels = shift_labels.reshape(n).to(torch.int32)
        grad_h = torch.empty((n, D), dtype=torch.float32, device=hidden.device)
        loss = torch.empty((n,), dtype=torch.float32, device=hidden.device)
        entropy = torch.empty((n,), dtype=torch.float32, device=hidden.device)
        hits = torch.empty((n,), dtype=torch.int32, device=hidden.device)
        BLOCK_V = 4096
        for start in range(0, n, ROWS):
            stop = min(start + ROWS, n)
            h16 = h[start:stop].to(weight16.dtype)
            logits = torch.matmul(h16, weight16.t())  # (rows, V) in the autocast dtype
            grad = torch.empty_like(logits)
            head_ce_kernel[(stop - start,)](logits, labels[start:stop], grad, loss[start:stop],
                                            entropy[start:stop], hits[start:stop],
                                            logits.stride(0), grad.stride(0), V,
                                            IGNORE=IGNORE_INDEX, BLOCK_V=BLOCK_V, num_warps=8)
            del logits
            grad_h[start:stop] = torch.matmul(grad, weight16)
            del grad
        inv = 1.0 / denominator
        grad_h.mul_(inv)
        total = loss.sum() * inv
        ctx.save_for_backward(grad_h)
        ctx.shape = (B, T, D)
        ctx.dtype = hidden.dtype
        labelled = labels != IGNORE_INDEX
        step_out.append(HeadStep(token_entropy=entropy.view(B, T), token_loss=loss.view(B, T),
                                 correct=hits.sum(), labelled=labelled.sum()))
        return total

    @staticmethod
    def backward(ctx, grad_loss):
        (grad_h,) = ctx.saved_tensors
        B, T, D = ctx.shape
        return (grad_h * grad_loss).view(B, T, D).to(ctx.dtype), None, None, None, None


class HeadHook:
    """Runs the causal LM's forward with the head held back and the loss built from the hidden
    states; a forward without labels is untouched. One instance per patched model class."""

    def __init__(self):
        self.last_step = None
        self._weight16 = None
        self._weight_key = None
        self.enabled = True
        self.dtype = torch.float16  # the matmul dtype of the head: what autocast uses on this card

    def weight16(self, lm_head):
        """The head's table in the matmul dtype: the parameter itself when it already is (see
        solution/embed.py), otherwise one cached cast."""
        w = lm_head.weight
        if w.dtype == self.dtype and w.is_contiguous():
            return w.detach()
        key = (w.data_ptr(), w.shape, w.dtype, w.device)
        if self._weight_key != key or self._weight16 is None:
            self._weight16 = w.detach().to(self.dtype).contiguous()
            self._weight_key = key
        return self._weight16

    def take_step(self):
        step, self.last_step = self.last_step, None
        return step


HOOK = HeadHook()


def _shift(labels):
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def patch_qwen2_causal_lm(dtype=torch.float16):
    HOOK.dtype = dtype
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.modeling_outputs import CausalLMOutputWithPast

    original = modeling_qwen2.Qwen2ForCausalLM.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        lm_head = getattr(self, "lm_head", None)
        if labels is None or not HOOK.enabled or not isinstance(labels, torch.Tensor) \
                or not labels.is_cuda or kwargs.get("shift_labels") is not None \
                or not isinstance(lm_head, torch.nn.Module) or logits_to_keep not in (0, None) \
                or not isinstance(getattr(lm_head, "weight", None), torch.Tensor):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        num_items = kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        shift_labels = _shift(labels).to(hidden.device)
        if num_items is None:
            denominator = (shift_labels != IGNORE_INDEX).sum().clamp(min=1).to(torch.float32)
        elif torch.is_tensor(num_items):
            denominator = num_items.to(device=hidden.device, dtype=torch.float32)
        else:
            denominator = torch.tensor(float(num_items), device=hidden.device)
        steps = []
        loss = FusedHeadCrossEntropy.apply(hidden, HOOK.weight16(lm_head), shift_labels,
                                           denominator, steps)
        HOOK.last_step = steps[0]
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    modeling_qwen2.Qwen2ForCausalLM.forward = forward


STEP_ATTR = "_soup_chunked_step"  # where Soup's rewind recorder looks for a step without logits


def patch_sft_trainer():
    """TRL's `SFTTrainer.compute_loss` reads `outputs.logits` for `entropy` and
    `mean_token_accuracy`. When our head ran (no logits), log the same columns by TRL's formulas
    from what the kernel collected, and leave the step for the rewind recorder."""
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not HOOK.enabled or not isinstance(inputs, dict) or "labels" not in inputs \
                or not isinstance(inputs["labels"], torch.Tensor) or not inputs["labels"].is_cuda:
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        HOOK.last_step = None
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        step = HOOK.take_step()
        if step is None:  # the hook did not run (another model class): TRL's own path
            if getattr(outputs, "logits", None) is None:
                raise RuntimeError("solution: the model returned no logits and no fused step")
            setattr(self, STEP_ATTR, None)
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        setattr(self, STEP_ATTR, step)
        metrics = getattr(self, "_metrics", None)
        gather = getattr(getattr(self, "accelerator", None), "gather_for_metrics", None)
        if metrics is not None and gather is not None:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
                tokens = gather(attention_mask.sum()).sum().item()
            else:
                entropy = step.token_entropy.mean()
                tokens = None
            metrics[mode]["entropy"].append(gather(entropy).mean().item())
            if mode == "train" and tokens is not None and hasattr(self, "_total_train_tokens"):
                self._total_train_tokens += tokens
            if hasattr(self, "_total_train_tokens"):
                metrics[mode]["num_tokens"] = [self._total_train_tokens]
            labelled = gather(step.labelled).sum()
            accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
            metrics[mode]["mean_token_accuracy"].append(accuracy)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss
