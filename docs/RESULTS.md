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

## Prefill (v3, client-side TTFT, cold unique prompts, max_tokens=1)

| prompt | TTFT | tok/s |
|---|---:|---:|
| ~8.3 K | 8.3–9.6 s | 870–1010 |
| ~33 K | 31.2 s | ~1066 |

Same ballpark as the E2 references (their server-side fully-uncached harness
reports ~1132 tok/s at 8K) — **no prefill regression** from the overlay: the
`reconstruct_hgemm` path (>144 rows) amortizes over prefill chunks.

## Draft-depth A/B on the v3 cost structure (k=5 vs k=7)

Hypothesis: with dense now cheap, the expert-read share (which scales with the
verify batch) might flip the k tradeoff. It does not:

| probe | k=7 | k=5 |
|---|---:|---:|
| structured | **72.6–73.0** | 58.9–59.1 |
| prose | 30.7–32.1 | 32.3–34.6 |
| code (fr) | **43.0–43.6** | 39.7–42.9 |

The acceptance loss on predictable content dominates the expert-read savings.
**k=7 stays.** (Prose mildly prefers shorter windows — consistent with what we
measured on DeepSeek-V4-Flash-Vision, where k=3 won on prose.)

## EXL3 draft (DFlash2 quantized to 5 bpw)

Draft converted with MiaAI-Lab/exllamav3 (1.4.2; DFlash2 is a first-class
architecture there), compiled inside the E2 image
(CPATH=<dist-packages>/nvidia/cu13/include for the slim toolkit). 860 MB vs
2.2 GB BF16, RMSE ~0.0009/linear. The draft has no embed/lm_head of its own
(shared with the target). Serving-side wiring:

- non_routed_exl3 keys at the draft's OFFSET runtime prefixes
  (`model.layers.45..49` + `model.fc` — start_layer_id = target layer count);
- the draft is built replicated with global tp attrs → the loader decides by
  SHAPES (narrow only when full == dest×tp);
- DFlash2 pre-builds a fused multi-layer KV weight by slicing
  `qkv_proj.weight[q_size:]` — gone once quantized (and the empty slice is
  silent). We reconstruct the K/V rows from the k/v trellis shards via
  chunked identity forwards (≤128 rows — beyond 144 rows LinearEXL3 switches
  to a CPU-unpacking reconstruct path that deadlocked TP ranks during
  dummy_run), triggered right after process_weights_after_loading.

| probe | BF16 draft | EXL3 draft |
|---|---:|---:|
| structured | 72.6–73.0 | **76.2–76.3** |
| prose (en) | 30.7–32.1 | 31.2–32.9 |
| code (fr) | 43.0–43.6 | **44.1–45.9** |
| code (en, BST) | — | **55.3** |
| draft acceptance (code) | 52–58 % | **52–55 %** |
| KV pool | 1.118 M | **1.152 M** |

## EXL3 lm_head (K6, shared target/draft head)

turboderp's 4.05bpw branch ships lm_head K6 (mul1). The head is a
`ParallelLMHead` (VocabParallelEmbedding), not a LinearBase — but its
quant-method interface is call-compatible with a linear method
(`create_weights(layer, dim, [vocab_per_rank], …)`, `apply(layer, x, bias)`),
so the same `Exl3LinearMethod` serves it: vocab-parallel = contiguous column
shard, `layer.tp_rank/tp_size` present. Because the DFlash2 draft shares the
target head, the BF16 head was read once per draft forward and once per
verify — quantizing it pays double.

| probe | EXL3 draft | + EXL3 lm_head |
|---|---:|---:|
| structured | 76.2–76.3 | **78.7–79.8** |
| prose (en) | 31.2–32.9 | 32.2–33.4 |
| code (fr) | 44.1–45.9 | **47.7–49.3** |
| code (en, BST) | 55.3 | **58.7** |
| KV pool | 1.152 M | **1.237 M** (= E2 baseline) |
