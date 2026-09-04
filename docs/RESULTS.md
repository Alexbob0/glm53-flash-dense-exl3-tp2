# Measurements

Hardware: 2× NVIDIA DGX Spark (GB10, sm_121, 128 GB unified LPDDR5X ~273 GB/s
each), ConnectX-7 point-to-point /30 (RoCEv2, single rail ~109 Gb/s).
Serving: MiaAI E2 image + this overlay, TP=2, 1M context, fp8_ds_mla KV,
DFlash2 k=7 draft, CUDA graphs, MAX_NUM_SEQS=4, MNBT 7168.
Protocol: streaming, temp 0, TTFT excluded, median of 3, `enable_thinking:false`
(same probes as MiaAI's `tests/bench_decode.py`; "code (fr)" is a French BST
implementation prompt, 400 tokens).

## End-to-end decode

| probe | E2 baseline | v1 dense (no KDA qkv) | v2 (+custom op, fp16) | v3 (+KDA q/k/v mixed) |
|---|---|---|---|---|
| structured | 65.0 | 68.2 | 68.1 | **72.6–73.0** |
| prose (en) | 26.2 | 28.2 | 30.5* | **30.7–32.1** |
| code (fr) | 38.4 | 39.2 | 38.4 | **43.0–43.6** |

\* prose is the noisiest probe (min–max spread up to ±2 tok/s).

Draft acceptance (DFlash2, code): ~52–58 % at every stage — the overlay does
not touch the drafter. KV pool: 1.24 M (E2) → 1.03 M (v1, fp32 shard outputs)
→ 1.07 M (v2) → 1.12 M (v3). Weight load: 79.8 GiB/rank (v2).

## Microbenchmarks (per-rank TP=2 shapes, m=1)

Weighted per-step totals over the covered dense modules:

| | exl3 trellis | cuBLAS BF16 |
|---|---|---|
| hot (single weight set — L2-inflated) | 5.9 ms | 16.0 ms |
| cold (4 rotating weight sets) | 5.4 ms | 17.2 ms |

Cold effective bandwidth: exl3 163–229 GB/s, BF16 ~160–170 GB/s. The kernel
is not the bottleneck at decode shapes (rows ≤ 32 → `bc.run_alloc` path).

## Torch-profiler breakdown of a real decode step

Window: 745 ms ≈ 8 steps (structured content), GPU busy 98 %. Per ~93 ms step
(v2 stack, KDA qkv still BF16):

| bucket | ms/step | share |
|---|---|---|
| `exl3_moe_kernel<4,256>` (routed experts) | ~45 | 48 % |
| cutlass BF16 (dense still BF16: KDA qkv, lm_head, …) | ~24 | 26 % |
| dense EXL3 (`exl3_gemm` K6/K5) | ~7.5 | 8 % |
| NCCL allreduce | ~7 | 8 % |
| KDA scan, mHC, elementwise, indexer, attention | ~9 | 10 % |

Interpretation: with k=7 speculative drafts, each verify step routes 8 tokens
→ up to ~64 distinct experts per MoE layer are read, ~17 GB/step; the E2
expert kernel runs at ~85 % of DRAM bandwidth. Dense quantization therefore
caps out: v3 converts most of the remaining BF16 (−~7 ms/step measured) and
the only dense lever left (lm_head, ~0.6 GB/rank) is worth ~+1 tok/s.
