# Soup Kernel Challenge: Студия Евгения Захарова

Triton kernels that make `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4 + LoRA r=16,
4 x 512 tokens, 64 steps) faster and much lighter on a free Kaggle T4, without changing what it
trains.

**Result on a Kaggle T4** (the kit's `harness/run_task.py`, every check of `harness/check_rules.py`
passing, see `results/`):

| | Base Soup 0.77.0 | This solution | Ratio |
|---|---|---|---|
| Whole `soup train` command (mean of 2 runs, order base, solution, solution, base) | 83.9 s | 50.2 s | **1.67x** faster |
| Peak GPU memory (nvidia-smi, max of the runs) | 14,345 MiB | 3,809 MiB | **3.77x** less |
| Held-out eval loss | 2.04965 | 2.04963 | -0.001% |
| Loss curve vs base | | first step 0.03%, mean 0.04% | |
| Own Triton kernels, share of training time | | 14.9% | |

`results/t4-run10-recompute14-2repeats`. The default configuration rebuilds the gate|up
activations of the last 14 decoder layers in the backward pass (`SOLUTION_RECOMPUTE=14`); the
same code with other counts, by the same protocol:

| `SOLUTION_RECOMPUTE` (layers rebuilt) | Time | Memory | Peak | Results folder |
|---|---|---|---|---|
| 0 | 1.64x | 3.20x | 4,489 MiB | `t4-run9-final-2repeats` |
| **14 (default)** | **1.67x** | **3.77x** | 3,809 MiB | `t4-run10-recompute14-2repeats` |
| 21 | 1.57x | 4.09x | 3,509 MiB | `t4-run12-recompute21-2repeats` |
| 24 | 1.56x | 4.23x | 3,389 MiB | `t4-run13-recompute24-2repeats` |
| 28 | 1.55x | 4.47x | 3,209 MiB | `t4-run11-recompute28-2repeats` |

Up to about half of the layers the recomputation is free: on Kaggle's two CPU cores the GPU idles
for part of every step waiting for kernel launches, and the extra GEMMs fill that idle time; past
that point the GPU becomes the bottleneck and the time grows. The base's wall time on a T4 varied
between 81.9 and 91.4 s across all our runs (±3% run to run), so differences of a few percent
between rows are within noise.

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
   are computed by the same formulas from what the kernel collected: TRL's `compute_loss` is
   patched, and the glue that logs the columns follows Soup's own
   `soup_cli/utils/stream_loss_chunk.py::_log_trl_metrics` (Apache-2.0) and leaves the step in
   the `_soup_chunked_step` slot the rewind recorder documents for a step without logits. Soup's
   `training.stream_loss_chunk` switch itself is not enabled (the config is the kit's); the
   kernel that does the work is ours. The idea is the fused linear cross-entropy of Liger Kernel /
   Unsloth; the code is ours.
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

7. **Padding removed inside the batch** (`packed.py`). TRL pads each batch to its longest row and
   with these rows 42% of the computed positions are padding. After the embedding the decoder
   gathers the real positions once and every layer, the LoRA branches and the output head work on
   the [1, N_real, D] rows. The attention runs PyTorch's SDPA causally per sequence on that
   sequence's rows: the same attention the base run computes through its causal + padding mask,
   minus the pad blocks the masked kernel still visits. Rows are never mixed across batches; the
   LoRA dropout masks are drawn on fp32 ones of the padded [B, T, D] shape, exactly as the base
   run draws them on its fp32 norm output (PyTorch's fused dropout maps its random numbers to
   elements by dtype), and gathered, so the masks are the base run's. This is the single largest
   speed-up.

8. **Kernels compiled while the model loads** (`warmup.py`). Every run starts with an empty
   Triton cache and on Kaggle's two CPU cores the first step paid about five seconds for the
   compilation of our kernels. `apply()` runs before Soup loads the model, so a background
   thread launches each kernel once on tiny tensors with the exact configuration training uses;
   the first real launch finds the compiled kernel. The compilation still happens inside the
   measured command, it just overlaps the model's load from disk.

Accompanying changes (no kernels, but they are where much of the memory went):

- **fp16 activations between the 4-bit matmuls** (`lowprec.py`). `prepare_model_for_kbit_training`
  makes the residual stream fp32, and bitsandbytes casts every projection's fp16 result back to
  fp32: with the base run, silu, the product and the LoRA add run in fp32 on fp16-exact values and
  the backward pass keeps fp32 copies. We keep the fp16 result (identical values), the residual
  adds stay fp32.
- **The embedding table in fp16** (`embed.py`). On a T4 Soup loads the frozen base in fp16
  (`soup_cli.utils.gpu.resolve_frozen_base_load_dtype`); PEFT's `prepare_model_for_kbit_training`
  then upcasts the tied embedding / `lm_head` table to fp32 (0.93 GB). We undo that upcast after
  the model is built: the table is bit-for-bit what Soup loaded, the head read it through fp16
  autocast anyway, the lookup still returns fp32 rows (the residual stream is unchanged), the
  adapter carries no embedding tensor and `eval_adapter.py` reloads the base itself.
- **MLP recompute in the last 14 layers** (`SOLUTION_RECOMPUTE=N`, default 14): the merged
  gate|up activations (the largest activation of the model) of the last N decoder layers are
  rebuilt in the backward pass from the saved input instead of being kept; our own autograd
  function, not Soup's `gradient_checkpointing` switch. See the table above for the trade-off.

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
`SOLUTION_RECOMPUTE=N` rebuild the gate|up activations of the last N layers in the backward pass
(default 14; 0 keeps everything, 28 rebuilds every layer);
`SOLUTION_NO_CE`, `SOLUTION_NO_RMSNORM`, `SOLUTION_NO_NF4`, `SOLUTION_NO_MLP`, `SOLUTION_NO_ATTN`,
`SOLUTION_NO_ROPE`, `SOLUTION_NO_LOWPREC`, `SOLUTION_NO_EMBED` (=1 turns that part off).

## Results and limits

`results/` holds the raw `result.json`, `checks.json`, per-run `metrics.json` and a kernel profile
of our Kaggle T4 runs: `t4-run1` (kernels only, 1.33x / 2.57x), `t4-run2-recompute` (1.18x / 3.74x),
`t4-run3-packed` (padding removed, one run each, with a kernel profile), `t4-run4-packed-recompute`
(the same code with `SOLUTION_RECOMPUTE=1`: 1.40x / 3.90x), `t4-run5-packed-2repeats` (the jury's
protocol, two runs per side: 1.50x / 2.96x), `t4-run6-rowattn-2repeats` (attention per sequence on
the packed rows, dropout masks as the base's: the table above), `t4-run7-recompute-final` (the
same code with `SOLUTION_RECOMPUTE=1`, two runs per side), `t4-run8-warmup-2repeats` (kernels
compiled in the background while the model loads), `t4-run9-final-2repeats` (the final code without
recompute), `t4-run10..13` (the final code with 14, 28, 21 and 24 layers rebuilt; 14 is the
default and the headline table).

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
- After the q|k|v and gate|up tensors are stacked, the original packed parameters release their
  storage (they keep their shape as zero-stride views); the training and the adapter never read
  them again, but a path that would (generation with a KV cache through the original PEFT
  forward) is not supported in this process.
- Our own kernels' share of the training time is 14.9% (lowest of two runs); without the two
  SwiGLU kernels (the operation the kit's example also fuses, written differently here, likeness
  0.60 by the kit's own measure) it stays above 10%, well over the 5% line.
- Not done: our own attention kernel (PyTorch's SDPA stays, per sequence) and a fused
  dequant-GEMM.

## Layout

| Path | What it is |
|---|---|
| `solution/__init__.py` | `apply()`: the patches, in order, with their guards |
| `solution/cross_entropy.py` | fused head + cross-entropy, TRL metrics, rewind step |
| `solution/quant.py` | NF4 unpacking kernel, merged weights, `QuantLinear` |
| `solution/attention.py`, `solution/rope.py`, `solution/packed.py` | merged q\|k\|v attention, rotary kernel, padding removed inside the batch |
| `solution/fused_mlp.py` | merged gate\|up MLP, gated SwiGLU kernels, optional recompute |
| `solution/rmsnorm.py`, `solution/lowprec.py`, `solution/embed.py` | RMSNorm kernel, fp16 activations, fp16 embedding |
| `solution/test_kernels.py` | every kernel against PyTorch / bitsandbytes, forward and backward |
| `results/` | Kaggle T4 measurements |
| `harness/`, `task/`, `tests/`, `templates/` | the organizers' kit, unchanged |

License: Apache-2.0 (the kit's), see `LICENSE` and `NOTICE`.
