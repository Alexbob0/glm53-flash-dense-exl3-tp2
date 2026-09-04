#!/usr/bin/env python3
"""Analyse d'une trace chrome torch-profiler vLLM : somme des durées des
kernels GPU par nom et par famille, sur la fenêtre stable de la trace."""
import gzip, json, sys, collections, re

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data

kern = [e for e in events
        if e.get("ph") == "X" and e.get("cat", "").lower() in ("kernel", "gpu_op")
        and "dur" in e]
if not kern:
    cats = collections.Counter(e.get("cat", "?") for e in events if e.get("ph") == "X")
    print("aucun kernel GPU trouvé; catégories:", dict(cats.most_common(10)))
    sys.exit(1)

kern.sort(key=lambda e: e["ts"])
t0, t1 = kern[0]["ts"], kern[-1]["ts"]
lo, hi = t0 + 0.15 * (t1 - t0), t0 + 0.85 * (t1 - t0)
win = [e for e in kern if lo <= e["ts"] <= hi]
wall = hi - lo

FAM = [
    ("nccl", r"nccl|AllReduce|ncclDevKernel"),
    ("exl3-trellis", r"exl3|trellis|hgemm_tp|bc_gemm"),
    ("moe-fat/E2", r"fat_gemm|moe|scatter|expert"),
    ("gemm-bf16", r"cutlass|gemv|s16816|ampere_|sm90|cublas|nvjet"),
    ("attention/mla", r"fmha|mla|attn|flash|paged"),
    ("indexer/topk", r"index|topk|top_k|compress"),
    ("kda/conv/scan", r"kda|conv|scan|chunk|recurrent|gated|delta"),
    ("norm/elementwise", r"norm|elementwise|vectorized|cat|copy|fill|reduce_kernel|softmax|rotary|silu|act"),
    ("sampler", r"sampl|gumbel|multinomial|argmax"),
]

by_name = collections.Counter()
for e in win:
    by_name[e["name"]] += e["dur"]
fam_tot = collections.Counter()
fam_of = {}
for name, dur in by_name.items():
    for fam, pat in FAM:
        if re.search(pat, name, re.I):
            fam_of[name] = fam
            break
    else:
        fam_of[name] = "autre"
    fam_tot[fam_of[name]] += dur

gpu_busy = sum(by_name.values())
print(f"fenêtre {wall/1e3:.1f} ms | GPU occupé {gpu_busy/1e3:.1f} ms ({100*gpu_busy/wall:.0f} %)")
print("\n-- par famille (ms, % du busy) --")
for fam, d in fam_tot.most_common():
    print(f"{fam:20s} {d/1e3:8.1f}  {100*d/gpu_busy:5.1f} %")
print("\n-- top 20 kernels --")
for name, d in by_name.most_common(20):
    print(f"{d/1e3:8.1f} ms  [{fam_of[name]:16s}] {name[:95]}")
