# Submission: Студия Евгения Захарова

## Links
- Repository, final commit of the code: https://github.com/BAITC-Hacks/hack-a8ca734b-solo-ai-team/commit/9667db8
  (the documentation commits after it change README.md / SUBMISSION.md only). Public mirror of the same
  commits, cloned by the notebook: https://github.com/fnc12/soup-kernel-challenge
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): https://www.kaggle.com/code/fnc123/soup-kernel-challenge?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Yevgeniy Zakharov | captain, everything | fnc12 | fnc123 | via the hackathon platform |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `head_ce_kernel` | `solution/cross_entropy.py` | the full-vocabulary logits, fp32 upcast, log-softmax, cross-entropy, TRL's entropy and accuracy passes, Soup's per-row losses | 0.8% |
| `nf4_dequant_kernel` | `solution/quant.py` | bitsandbytes' two-kernel NF4 unpacking (absmax rebuild + dequant) in the forward and the backward pass | 7.7% |
| `swiglu_gated_kernel`, `swiglu_gated_bwd_kernel` | `solution/fused_mlp.py` | silu, product and their backward on the merged gate\|up buffer | 1.9% + 3.1% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm's five fp32 ops and three saved copies | 0.6% + 0.7% |
| `rope_kernel` | `solution/rope.py` | slice / negate / concatenate / multiply / add of the rotary embedding for q and k, forward and backward | 2.7% |

(shares from `results/t4-run3-packed/solution_1_metrics.json`; the final revision measures 14.9%
together, `results/t4-run6-rowattn-2repeats`)

Ideas taken from other kernels, with links (we wrote the code ourselves):
- fused linear cross-entropy in row chunks: Liger Kernel
  (https://github.com/linkedin/Liger-Kernel) and Unsloth (https://github.com/unslothai/unsloth);
- NF4 layout and the double-quantized absmax: bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes),
  our kernel reproduces its arithmetic;
- the metric glue for a step without logits follows Soup's own
  `soup_cli/utils/stream_loss_chunk.py::_log_trl_metrics` (Apache-2.0), so TRL's columns and
  `rewind.jsonl` use the same formulas.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s (mean of 2 runs) | 85.73 | 53.65 | speedup 1.60x |
| Peak GPU memory, MiB | 14,345 | 3,553 | base/ours 4.04x |
| Eval loss (public rows) | 2.04967 | 2.04962 | -0.002% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (the profile: casts, the vocabulary logits, padding), how
each kernel works, the merged weights and the padding-free decoder, results against the base
(speed, memory, quality, all checks), limits and next steps.

## Limitations
- Attention is PyTorch's SDPA (per sequence on the packed rows), not our kernel; a fused
  dequant-GEMM was not attempted (cuBLAS wins on a T4 for these shapes).
- The patches target Qwen2 + NF4 (blocks of 64, double quantization) + PEFT's vanilla LoRA;
  other layouts fall through to the original code.
- `SOLUTION_RECOMPUTE=N` (default 14): rebuilding the gate|up activations of more layers buys memory for time (README has the table).
- The NF4 weights and the saved activations of the decoder wait in pinned host memory between uses
  (`SOLUTION_STREAM`, `SOLUTION_OFFLOAD`, both on by default): the same bytes, nothing recomputed or
  approximated; the copies ride the PCIe bus on their own CUDA streams while the GPU computes. The
  frozen fp16 embedding / head table (467 MB) stays on the GPU: streaming it through the head would
  cost more time than its memory is worth under the scoring.
- Soup's own switches (`gradient_checkpointing`, `activation_offloading`, ...) are untouched; the
  recompute and the offloading above are our own autograd functions and saved-tensor hooks.
- Next: a Triton attention kernel over the packed rows, and the NF4 unpacking fused into the
  GEMM's operand load.
