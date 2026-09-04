# GLM-5.3-Flash: dense-EXL3 overlay under TP=2 on 2× DGX Spark (GB10)

Serving-side work that quantizes the **non-expert linears** of
GLM-5.3-Flash (attention, shared experts, dense MLPs) to EXL3 K6/K5 on top of
the routed-experts-only EXL3 pack, under **tensor parallelism TP=2** on two
NVIDIA DGX Spark (GB10, sm_121) — including, as far as we know, the first
published TP-sharded loader for a **merged linear that mixes EXL3 shards,
TP-sharded BF16 shards and replicated BF16 shards in one module** (GLM's fused
KDA projection `in_proj_qkvbfg_a`).

**Measured end to end** (streaming decode, TTFT excluded, temp 0, median of 3,
two boots):

| probe | baseline (E2 stack) | + dense overlay | + KDA q/k/v (this work) | Δ |
|---|---|---|---|---|
| structured (count 1-200) | 65.0 tok/s | 68.2 | **72.6–73.0** | **+12 %** |
| prose (hashmap, en) | 26.2 | 28.2 | **30.7–32.1** | **+17 %** |
| code (fr) | 38.4 | 39.2 | **43.0–43.6** | **+12 %** |

Output quality unchanged (coherent French code — a sharding mistake in
attention garbles output instantly), DFlash2 draft acceptance unchanged
(~52–58 % on code), KV pool 1.12 M tokens at 1M context.

## Why this is worth reading even if you don't run GLM

The measurement story is the real contribution. Three things everyone assumes
turned out to be false on this workload, and we have the receipts:

1. **"Fewer weight bytes ⇒ proportionally faster decode" — false here.**
   Microbenchmarks (hot *and* L2-busted cold) show the exl3 trellis GEMV at
   163–229 GB/s effective, promising −10–12 ms/step over BF16. End to end we
   recovered a fraction of that at first. The missing time was not the kernel,
   not the glue, not dynamo graph breaks (a custom-op variant changed nothing).
2. **A torch-profiler trace of a real decode step** (`tools/analyze_trace.py`)
   gave the actual budget on a ~93 ms step:
   ~45 ms (48 %) `exl3_moe_kernel` — **routed experts, amplified by
   speculative verification** (k=7 drafts make each step read up to ~64
   distinct experts per MoE layer instead of 8; the kernel already runs at
   ~85 % of DRAM bandwidth, nothing to optimize) · ~24 ms remaining-BF16 dense
   · ~7.5 ms our EXL3 denses (exactly what the microbench predicted) ·
   ~7 ms NCCL allreduce · ~9 ms everything else.
3. **Under a MoE with speculative decoding, dense-weight quantization has a
   hard ceiling**: converting the remaining dense BF16 (this repo's v3) buys
   +12 %, and the next lever (lm_head) is worth ~+1 tok/s. The wall is expert
   read traffic under the verify batch — structural, not a config or kernel
   issue.

## What's in here

- `overlay/exl3.py` — drop-in replacement for the quantization plugin of
  [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
  (their E2 fat-expert version), adding:
  - `non_routed_exl3` pack-config support (per-module bits, mixed codebooks
    mcg/mul1),
  - `Exl3LinearMethod`: dense EXL3 linears with **per-shard TP geometry** —
    the loader reads `layer.tp_rank/tp_size` (so `disable_tp` replicated
    layers such as the fused MLA a-projection load unsharded) and
    `layer.replicated_shard_ids` (so the KDA merged projection loads its six
    heterogeneous shards correctly: q/k/v EXL3 column-sharded, `b` BF16
    column-sharded, `f_a`/`g_a` BF16 replicated). Note the layer's own
    forced-rank weight-loader trick is bypassed once params carry a custom
    `weight_loader` — the closure has to reimplement it.
  - a torch **custom op** wrapping the per-layer forward (opaque to dynamo,
    cudagraph-friendly), fp16 end to end.
- `patches/` — two ~10-line diffs the fork needs so dense modules actually
  receive the quant config (MLA layers were built with `quant_config=None`;
  the KDA wrapper froze `self.quant_config=None`).
- `Dockerfile.dense` — derived image: E2 base + the three files above.
- `tools/` — pack cache-entry builder (relative symlinks — absolute ones
  dangle inside the container), the two microbenchmarks, the trace analyzer.

The overlay **pack** itself is built with
[vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3)'s
`tools/dense_overlay.py` (byte-range reads from
[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
branch `4.05bpw` over
[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)):

```
dense_overlay.py --branch 4.05bpw --src <TR3-4bpw snapshot> --out <overlay dir> \
    --prefix-rewrite model.language_model.:language_model.model.
```

The prefix rewrite matters: the serving model is
`Glm5NextForConditionalGeneration` (vision on), so runtime module prefixes are
`language_model.model.layers.N...`, and the config keys are matched **exactly**.

## Reproducing

1. Build the overlay pack (above), `--verify`.
2. `tools/make-cache-entry.sh` to expose it as a local HF-cache model.
3. Build the derived image: `docker build -f Dockerfile.dense -t glm53-dense:k6 .`
   (identically on both nodes; compare the three files by sha256, not image ID).
4. Point the MiaAI launcher at it (`IMAGE=glm53-dense:k6`,
   `MODEL_CACHE_NAME=<entry>`, `SKIP_BUILD=1 SKIP_SHIP=1`, distinct container
   names) and start.

Serving profile used for the numbers: TP=2, 1M context, fp8_ds_mla KV,
DFlash2 k=7 draft (TP=2), CUDA graphs on, `MAX_NUM_SEQS=4`, MNBT 7168.

## Pitfalls we hit (so you don't)

See [docs/PITFALLS.md](docs/PITFALLS.md) — profiler windows that kill the
server, the container-side trace directory, sourced-env quoting, OOM when
microbenching next to a live engine at util 0.87, and more.

## Credits & licenses

Derived from and building on: **MiaAI-Lab** GLM-5.3-Flash EXL3 recipe and E2
kernels (MIT) · **vcruz305/vllm-exl3** non-routed EXL3 design, validated TP=1
(Apache-2.0) · **turboderp** GLM-5.3-Flash-exl3 quants and ExLlamaV3 ·
**Mia-AiLab** TR3-4bpw pack. This repo: MIT (see NOTICE for the Apache-2.0
derived parts). Weight licenses are those of their respective packs — this
repo redistributes **no weights**.
