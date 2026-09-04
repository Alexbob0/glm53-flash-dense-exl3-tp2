# GLM-5.3-Flash — dense-EXL3 overlay, TP=2, 2× DGX Spark (GB10)

Quantizes the **non-expert linears** of GLM-5.3-Flash (attention, shared
experts, dense MLPs — K6/K5 EXL3) on top of the routed-experts-only TR3-4bpw
pack, served in **TP=2** across two DGX Sparks with the
[MiaAI-Lab E2 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
Includes what we believe is the first published TP-sharded loader for a merged
linear mixing **EXL3 shards + TP-sharded BF16 + replicated BF16 in one module**
(GLM's fused KDA projection `in_proj_qkvbfg_a`).

No weights are redistributed — the overlay pack is built locally by
byte-range reads from turboderp's quants (~5.3 GB).

## Decode (this kit, 2026-09-04) ✅

Streaming, temp 0, TTFT excluded, median of 3, thinking off, two boots.
Baseline = same kit on the stock E2 stack, same protocol.

| probe | E2 baseline | **+ dense overlay (this repo)** | Δ |
|---|---:|---:|---:|
| structured (count 1→200) | 65.0 | **72.6–73.0** | **+12 %** |
| prose (hash-map, en) | 26.2 | **30.7–32.1** | **+17 %** |
| code (fr, BST impl) | 38.4 | **43.0–43.6** | **+12 %** |

Output quality unchanged (an attention-sharding mistake garbles output
instantly — it doesn't), DFlash2 acceptance unchanged (~52–58 % on code),
KV pool **1,118,466** tokens at 1M, weight load **79.8 GiB/rank**.

```bash
# structured (count 1→200) / prose (hash-map) / code (fr) — MiaAI bench_decode.py protocol
curl -s http://HEAD:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"GLM-5.3-Flash-EXL3-dense","temperature":0,"stream":true,"max_tokens":200,
  "messages":[{"role":"user","content":"Count from 1 to 200. Output only the numbers, separated by spaces. No other text."}],
  "chat_template_kwargs":{"enable_thinking":false}}'
```

## Step anatomy — why dense quantization caps out here (2026-09-04) ✅

Torch-profiler trace of a live decode step (745 ms window ≈ 8 steps, GPU busy
98 %, before the KDA q/k/v conversion — see
[docs/RESULTS.md](docs/RESULTS.md) for the full method):

| bucket | ms / ~93 ms step | share |
|---|---:|---:|
| `exl3_moe_kernel` — routed experts | ~45 | **48 %** |
| cutlass BF16 — dense not yet converted | ~24 | 26 % |
| dense EXL3 (`exl3_gemm` K6/K5, this repo) | ~7.5 | 8 % |
| NCCL allreduce | ~7 | 8 % |
| KDA scan · mHC · elementwise · indexer · attn | ~9 | 10 % |

⚠️ **The wall is structural**: with DFlash2 k=7, each verify step routes 8
draft tokens → up to ~64 distinct experts *per MoE layer* are read (~17
GB/step), and the E2 expert kernel already runs at ~85 % of DRAM bandwidth.
Dense-side EXL3 delivers exactly what microbenchmarks promise (hot **and**
L2-busted cold: 163–229 GB/s effective, ~3× fewer bytes than BF16), but it can
only claim the dense slice of the step. Converting the remaining lm_head is
worth ~+1 tok/s; beyond that, this axis is done.

## What runs

| Layer | Runtime |
|---|---|
| Base image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` (E2 fat-expert, recipe stamp `e2fe85b0…`) |
| This image | `glm53-dense:k6` = base + 3 files (`Dockerfile.dense`) — build on **each** node, compare the 3 files by sha256, never by image ID |
| vLLM | `v0.1.dev20051+g487ecf187` (MiaAI fork, in-image) |
| ExLlamaV3 | `0.0.43` @ `c5d9c657` (in-image; `LinearEXL3` accepts `mcg` and `mul1`) |
| Base pack | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ `25a44fdb` (routed experts EXL3 K4/MCG) |
| Overlay source | `turboderp/GLM-5.3-Flash-exl3` branch **`4.05bpw`** (attn K6, shared K6, dense MLP K5 — mul1 codebook) |
| Overlay pack | 1,260 tensors / 5.34 GB, **191 config keys** incl. 34 × `in_proj_qkvbfg_a` with `bf16_shards:[3,4,5]` |
| Dense in EXL3 | MLA q_a/kv_a/q_b/o_proj · KDA **q/k/v** + o_proj · shared gate_up/down · dense MLP (layers 0-2) |
| Still BF16 | KDA `b`/`f_a`/`g_a` (mixed shards of in_proj) · lm_head · embeddings · indexer · MTP layer 45 |
| Spec / KV / ctx | DFlash2 k=7 (draft TP=2) · `fp8_ds_mla` · 1M (`MAX_NUM_SEQS=4`, MNBT 7168, util 0.87) |
| Serve profile | vision ON (`Glm5NextForConditionalGeneration`) → module prefixes `language_model.model.layers.N…` |

## Build & run

```bash
# 1. overlay pack (byte-range download ~5.3 GB, then verify)
python3 dense_overlay.py --branch 4.05bpw \
  --src <TR3-4bpw snapshot dir> --out <overlay dir> \
  --prefix-rewrite model.language_model.:language_model.model.        # then add --verify
# dense_overlay.py is vcruz305/vllm-exl3 `tools/` — see Credits

# 2. expose it as a local HF-cache model (RELATIVE symlinks — see Pitfalls)
HF_CACHE=~/hf SRC_SNAPSHOT=<snapshot> OVERLAY=<overlay dir> tools/make-cache-entry.sh

# 3. derived image, on BOTH nodes
docker build -f Dockerfile.dense -t glm53-dense:k6 .

# 4. MiaAI launcher, .env deltas:
#    IMAGE=glm53-dense:k6   MODEL_CACHE_NAME=models--local--glm53-dense-K6
#    SKIP_BUILD=1 SKIP_SHIP=1   CONTAINER_HEAD/WORKER=<distinct names>
./start.sh
```

Boot proof it's live: one
`[dense-overlay] EXL3 dense linear active (K=…, custom op)` log line **per
configured module** (expect the full count — some fork constructors pass
`quant_config=None` and silently skip you otherwise; that's what the two
~10-line diffs in [`patches/`](patches/) fix).

## What the loader adds (overlay/exl3.py)

- `non_routed_exl3` pack-config: exact-prefix per-module `{bits, bf16_shards}`.
- `Exl3LinearMethod` with **per-shard TP geometry**: reads
  `layer.tp_rank/tp_size` (so `disable_tp` replicated layers — the fused MLA
  a-projection — load unsharded) and `layer.replicated_shard_ids` (so the KDA
  merged projection loads six heterogeneous shards: q/k/v EXL3
  column-sharded, `b` BF16 column-sharded, `f_a`/`g_a` BF16 replicated).
  ⚠️ Once a param carries a custom `weight_loader`, the layer's own
  forced-rank trick is bypassed — the closure reimplements it.
- Per-layer forward wrapped as a **torch custom op** (opaque to dynamo,
  cudagraph-friendly), fp16 end to end; mixed BF16 shards run `F.linear`
  slices concatenated in shard order.
- mul1 **and** mcg codebook markers (experts are mcg, turboderp denses are mul1).

## Pitfalls

Paid for so you don't have to — profiler windows that kill the engine,
container-side trace dirs, sourced-env quoting, L2-flattered microbenches,
index-vs-safetensors stripping: [docs/PITFALLS.md](docs/PITFALLS.md).

## Credits & licenses

**MiaAI-Lab** — GLM-5.3-Flash EXL3 recipe + E2 kernels (MIT) ·
**vcruz305/vllm-exl3** — non-routed EXL3 design, TP=1 (Apache-2.0) ·
**turboderp** — ExLlamaV3 + GLM-5.3-Flash-exl3 quants ·
**Mia-AiLab** — TR3-4bpw pack. This repo: MIT, see [NOTICE](NOTICE).
Weight licenses are those of their respective packs.
