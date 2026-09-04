#!/usr/bin/env python3
"""Microbench exl3_gemm dense vs cuBLAS BF16 aux shapes per-rank TP=2 de la
pile glm53-dense (étape 0 du chantier kernel). Timing par CUDA events,
contenu aléatoire (timing indépendant des valeurs)."""
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.exl3 import (
    make_linear_exl3, MUL1_MARKER_SIGNED_INT32,
)

# (nom, in_per_rank, out_per_rank, K, occurrences par step)
SHAPES = [
    ("kda.o_proj      ", 4096, 4096, 6, 34),
    ("mla.q_a (repl)  ", 4096, 1536, 6, 11),
    ("mla.kv_a (repl) ", 4096, 512, 6, 11),
    ("mla.q_b         ", 1536, 8192, 6, 11),
    ("mla.o_proj      ", 4096, 4096, 6, 11),
    ("shared.gate_up  ", 4096, 2048, 6, 42),
    ("shared.down     ", 1024, 4096, 6, 42),
    ("mlp.gate_up     ", 4096, 12288, 5, 3),
    ("mlp.down        ", 6144, 4096, 5, 3),
]

def time_fn(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # µs

def main():
    dev = "cuda"
    mul1 = torch.tensor([MUL1_MARKER_SIGNED_INT32], dtype=torch.int32, device=dev)
    for m in (1, 4, 8):
        print(f"\n=== batch m={m} ===")
        tot_exl3 = tot_bf16 = 0.0
        print(f"{'module':17s} {'exl3 µs':>8s} {'bf16 µs':>8s} {'ratio':>6s} "
              f"{'exl3 GB/s':>9s} {'bf16 GB/s':>9s}")
        for name, k_in, n_out, K, cnt in SHAPES:
            trellis = torch.randint(-32768, 32767,
                                    (k_in // 16, n_out // 16, 16 * K),
                                    dtype=torch.int16, device=dev)
            suh = torch.randn(k_in, dtype=torch.float16, device=dev)
            svh = torch.randn(n_out, dtype=torch.float16, device=dev)
            lin = make_linear_exl3(trellis, suh, svh, None, mul1,
                                   out_dtype=torch.float16)
            x16 = torch.randn(m, k_in, dtype=torch.float16, device=dev)
            xbf = torch.randn(m, k_in, dtype=torch.bfloat16, device=dev)
            wbf = torch.randn(n_out, k_in, dtype=torch.bfloat16, device=dev)
            t_e = time_fn(lambda: lin.forward(x16, {}, out_dtype=torch.float16))
            t_b = time_fn(lambda: F.linear(xbf, wbf))
            by_e = k_in * n_out * K / 8
            by_b = k_in * n_out * 2
            print(f"{name} {t_e:8.1f} {t_b:8.1f} {t_e/t_b:6.2f} "
                  f"{by_e/t_e/1e3:9.1f} {by_b/t_b/1e3:9.1f}")
            tot_exl3 += t_e * cnt
            tot_bf16 += t_b * cnt
        print(f"{'TOTAL/step (somme ponderee)':30s} exl3 {tot_exl3/1000:.2f} ms "
              f"vs bf16 {tot_bf16/1000:.2f} ms (delta {(tot_exl3-tot_bf16)/1000:+.2f} ms)")

if __name__ == "__main__":
    main()
