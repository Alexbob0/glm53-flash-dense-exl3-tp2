# Pitfalls, in the order we paid for them

**Loader / plugin**

- `@register_quantization_config` must sit immediately above the class —
  inserting a helper between decorator and class hands the decorator a
  function (`issubclass() arg 1 must be a class`). The MiaAI launcher's GPU
  self-check catches this before anything serves; keep that check on.
- Once a parameter carries a custom `weight_loader`, the **layer's** loader
  logic is bypassed entirely — including
  `_Glm5NextMergedColumnParallelLinear`'s forced-`tp_rank=0` trick for
  replicated shards. Your closure must reimplement per-shard TP geometry:
  read `layer.replicated_shard_ids` and `layer.tp_rank/tp_size` (the latter
  also makes `disable_tp` layers — e.g. `DeepSeekV2FusedQkvAProjLinear` —
  load unsharded without special-casing).
- Config keys in `non_routed_exl3.layers` are matched **exactly** against
  runtime module prefixes. With the vision tower enabled the tree is
  `language_model.model.layers.N...` — derive the prefix from the model
  class actually instantiated, not from the checkpoint tensor names.
- The fork builds some dense modules with `quant_config=None` (MLA layers) or
  freezes `self.quant_config=None` in a constructor wrap (KDA). Without the
  two patches in `patches/`, those modules silently stay unquantized —
  coverage looks fine until you count activation log lines (expect one per
  configured module).
- `LinearEXL3.forward` auto-switches at `AUTO_RECONSTRUCT_THRESHOLD=144`
  rows: decode verify batches take the fast trellis path, prefill takes
  reconstruct+hgemm. Don't benchmark the wrong path.

**Measuring**

- A microbench looping on one weight set measures **L2, not DRAM** (we saw
  360 GB/s "effective" on a 273 GB/s part). Rotate several weight copies.
- Don't launch a GPU microbench next to a live engine at
  `gpu_memory_utilization=0.87` — instant OOM. (Unified-memory GB10.)
- vLLM torch profiler (`--profiler-config
  '{"profiler":"torch","torch_profiler_dir":...}'`):
  - keep the profiled window **short** (tens of tokens). A ~500-token window
    produced a trace whose export outlived the internal RPC timeout —
    `Call to profile method failed: cancelled`, server gone.
  - chain warmup → start → gen → stop without idle gaps; we lost one engine
    to a cross-rank shm-broadcast `TimeoutError` on the first request after
    ~25 min idle.
  - `stop_profile` can outlive your HTTP timeout while the server keeps
    exporting; poll the trace directory instead of trusting the status code.
  - traces land in the **container's** `torch_profiler_dir` — check
    `docker inspect` mounts for where that really is on the host.
- If your launcher `source`s an env file, an inline JSON arg needs single
  quotes *and* no spaces (word-splitting later): 
  `EXTRA_ARGS='--profiler-config {"profiler":"torch",...}'`.

**Packs & cache**

- Symlinks in a local pack must be **relative**: the serving container mounts
  the HF cache at a different absolute path, absolute symlinks dangle there.
- Removing tensors from the index does not remove them from safetensors
  files; vLLM iterates file contents. To drop modules from an overlay,
  rewrite the safetensors (the format is 8-byte header length + JSON header +
  raw bytes — pure-python is enough), fix the index, and restore the
  `.weight` entries of anything you reverted to BF16.
- Verify images across nodes by **content** (sha256 the changed files), never
  by image ID — locally built layers embed timestamps.
