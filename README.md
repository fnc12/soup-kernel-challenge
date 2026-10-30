# Soup Kernel Challenge: Solo AI Team

Triton kernels that make `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 + LoRA r=16,
4 x 512 tokens, 64 steps) faster and much lighter on a free Kaggle T4, without changing what it
trains.

**Result on a Kaggle T4** (the kit's `harness/run_task.py`, every check of `harness/check_rules.py`
passing, see `results/`):

| | Base Soup 0.77.0 | This solution | Ratio |
|---|---|---|---|
| Whole `soup train` command | 91.2 s | 68.7 s | **1.33x** faster |
| The 64 training steps | 56.5 s | 46.2 s | 1.22x |
| Peak GPU memory (nvidia-smi) | 14,345 MiB | 5,591 MiB | **2.57x** less |
| Held-out eval loss | 2.04964 | 2.04948 | -0.008% |
| Loss curve vs base | | first step 0.01%, mean 0.04% | |
| Own Triton kernels, share of training time | | 14.0% | |

Kaggle notebook: _link to be added at submission_ (`SUBMISSION.md` carries the final commit).

## What we did

Where a base step on the T4 goes (organizers' profile): 36% matmuls, 44% element-wise ops and
dtype casts, 7% softmax / cross-entropy over the 151,936-token vocabulary, 5% attention, 5% NF4
unpacking. The memory (14.3 of 15 GB) goes to the full-vocabulary logits and their fp32 copies,
fp32 copies of every projection's output, and an fp32 embedding table nobody reads in fp32.

Everything lives in `solution/`; `solution.apply()` patches, narrowly and with guards, the model
classes, bitsandbytes' 4-bit linear, TRL's `compute_loss` and Soup's model setup. Kernels, all
`@triton.jit` in `solution/`:

1. **Fused output head + cross-entropy in row chunks** (`cross_entropy.py`, `head_ce_kernel`).
   The head's fp16 matmul runs 256 positions at a time and one kernel does, in a single
   read-and-write of that chunk of logits, everything the base run does with the whole
   [4 x 512 x 151,936] tensor: log-sum-exp, the loss at the label, the gradient of the logits,
   the entropy of every position and the argmax hit for the accuracy. The gradient on the hidden
   states is taken chunk by chunk, so no logits outlive their chunk (the base keeps about 4 GB of
   them: fp16 logits, fp32 upcast, log-softmax, and TRL's and Soup's extra passes). The
   `entropy`, `mean_token_accuracy`, `num_tokens` columns and the per-row losses of `rewind.jsonl`
   are computed by the same formulas from what the kernel collected (TRL's `compute_loss` is
   patched; the rewind recorder reads the step from Soup's own `_soup_chunked_step` slot). The
   idea is the fused linear cross-entropy of Liger Kernel / Unsloth; the code is ours.
2. **NF4 unpacking in one pass** (`quant.py`, `nf4_dequant_kernel`). For this batch size
   bitsandbytes does not use its fused GEMM: each projection first rebuilds the double-quantized
   absmax (one kernel), then unpacks the weight (another), then calls cuBLAS, and the backward
   pass does it again. Our kernel reads the packed nibbles once, applies both quantization levels
   and writes the dense fp16 rows with one coalesced store (`tl.join`). `QuantLinear` is the
   autograd function around it (the weights are frozen: only the input needs a gradient).
3. **Merged q|k|v and gate|up weights** (`quant.QuantWeight`, `attention.py`, `fused_mlp.py`).
   The packed NF4 tensors of q, k, v (and of gate, up) are stacked once into one [N, K] weight
   with a per-scale-block offset array, so each pair/triple is one unpacking and one GEMM instead
   of three narrow ones (k and v are only 256 wide, a poor shape for tensor cores). The originals
   release their storage. PEFT's LoRA branches on q and v are computed with PEFT's own modules in
   the same order, so the dropout masks are the same as in the base run.
4. **2-D gated SwiGLU over the merged buffer** (`fused_mlp.py`, `swiglu_gated_kernel`,
   `swiglu_gated_bwd_kernel`): reads gate and up as the two halves of each row of the merged
   gate|up output, writes `silu(gate) * up` in fp16; its backward writes both gradients into one
   [R, 2I] buffer, exactly what the merged GEMM's backward consumes.
5. **Fused RMSNorm** (`rmsnorm.py`): one kernel forward (fp32 statistics, fp16 output), one
   backward recomputing from the saved input and one scalar per row; the norm weights are frozen,
   so no weight gradient is accumulated.
6. **Rotary embedding in one kernel each way** (`rope.py`): replaces the slice / negate /
   concatenate / multiply / add chain for q and k (and its backward) with one program per position.

Accompanying changes (no kernels, but they are where much of the memory went):

- **fp16 activations between the 4-bit matmuls** (`lowprec.py`). `prepare_model_for_kbit_training`
  makes the residual stream fp32, and bitsandbytes casts every projection's fp16 result back to
  fp32: with the base run, silu, the product and the LoRA add run in fp32 on fp16-exact values and
  the backward pass keeps fp32 copies. We keep the fp16 result (identical values), the residual
  adds stay fp32.
- **The embedding table in fp16** (`embed.py`). The frozen, tied embedding / `lm_head` table
  (0.93 GB in fp32) is kept in the compute dtype: every bf16 checkpoint value with |v| >= 2^-14 is
  exact in fp16, the head read it through fp16 autocast anyway, and the lookup still returns fp32
  rows, so the residual stream is unchanged.
- **Optional MLP recompute** (`SOLUTION_RECOMPUTE=1`): the merged gate|up activations (the largest
  activation: 28 x 4 x 512 x 17,920 x 2 bytes = 2 GB) are rebuilt in the backward pass from the
  saved input instead of being kept; our own autograd function, not Soup's
  `gradient_checkpointing` switch. See the results table for the trade-off.

What stays exactly as in the base run: the model, data, batches, steps, LoRA (r, alpha, dropout,
targets), optimizer and schedule, every Soup switch, the adapter's tensors, Soup's outputs
(metrics, `rewind.jsonl`). The regression tests (`SOUP_APPLY_SOLUTION=1 python -m pytest tests`,
79 tests) pass with the patches; every patch checks that it is looking at a real CUDA module and
falls through to the original code otherwise.

## How to run

Kaggle (the jury's way): import `templates/kaggle_notebook.ipynb`, accelerator **GPU T4 x2**,
internet on, put this repository's URL in the first cell, **Run All** (about 10 minutes). It
trains the base and the solution, scores both adapters and runs the checks.

Locally (any CUDA GPU; only T4 numbers count, on Ampere+ Soup trains in bf16):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q          # our kernels against plain PyTorch / bitsandbytes
python harness/run_task.py            # base, then solution -> runs/result.json
python harness/check_rules.py         # the ten checks -> runs/checks.json
```

Switches (environment variables, for ablations; all off by default except where noted):
`SOLUTION_RECOMPUTE=1` recompute the MLP's gate|up in the backward pass;
`SOLUTION_NO_CE`, `SOLUTION_NO_RMSNORM`, `SOLUTION_NO_NF4`, `SOLUTION_NO_MLP`, `SOLUTION_NO_ATTN`,
`SOLUTION_NO_ROPE`, `SOLUTION_NO_LOWPREC`, `SOLUTION_NO_EMBED` (=1 turns that part off).

## Results and limits

`results/` holds the raw `result.json`, `checks.json` and per-run `metrics.json` of our Kaggle T4
runs (the table above is `results/t4-run1`).

Limits and honest notes:

- The NF4 kernel reproduces bitsandbytes' values up to one fp16 ulp on a few values per million
  (bitsandbytes rounds the second-level absmax once more than our fused multiply-add). The loss
  curve stays within 0.04% of the base.
- The fused head computes the entropy and the per-row losses in fp32 from fp16 logits; TRL's
  `entropy_from_logits` and Soup's `row_losses` work in the logits' dtype. Differences are far
  inside the 3% tolerance (check 5 passes).
- The merged weights are built on the first forward pass (about 0.4 s of unpacking and
  concatenation), and the Triton kernels compile on the first steps; both are inside the measured
  time.
- The patches target Qwen2 with NF4 (blocks of 64, double quantization) and PEFT's vanilla LoRA;
  any other layout falls through to the original code paths.
- Not done: our own attention (SDPA's memory-efficient kernel stays), a fused dequant-GEMM, and
  padding removal inside the batch.

## Layout

| Path | What it is |
|---|---|
| `solution/__init__.py` | `apply()`: the patches, in order, with their guards |
| `solution/cross_entropy.py` | fused head + cross-entropy, TRL metrics, rewind step |
| `solution/quant.py` | NF4 unpacking kernel, merged weights, `QuantLinear` |
| `solution/attention.py`, `solution/rope.py` | merged q\|k\|v attention, rotary kernel |
| `solution/fused_mlp.py` | merged gate\|up MLP, gated SwiGLU kernels, optional recompute |
| `solution/rmsnorm.py`, `solution/lowprec.py`, `solution/embed.py` | RMSNorm kernel, fp16 activations, fp16 embedding |
| `solution/test_kernels.py` | every kernel against PyTorch / bitsandbytes, forward and backward |
| `results/` | Kaggle T4 measurements |
| `harness/`, `task/`, `tests/`, `templates/` | the organizers' kit, unchanged |

License: Apache-2.0 (the kit's), see `LICENSE` and `NOTICE`.
