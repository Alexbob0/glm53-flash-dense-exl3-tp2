#!/usr/bin/env python3
"""Variante FROID : rotation de 4 jeux de poids par shape pour casser le L2.
Mesure la bande passante DRAM effective réelle du kernel trellis vs cuBLAS."""
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.exl3 import (
    make_linear_exl3, MUL1_MARKER_SIGNED_INT32,
)

SHAPES = [
    ("kda.o_proj   ", 4096, 4096, 6, 34),
    ("mla.q_b      ", 1536, 8192, 6, 11),
    ("shared.gateup", 4096, 2048, 6, 42),
    ("shared.down  ", 1024, 4096, 6, 42),
    ("mlp.gate_up  ", 4096, 12288, 5, 3),
]
NCOPIES = 4

def time_rot(fns, iters=200, warmup=40):
    for i in range(warmup): fns[i % NCOPIES]()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(iters): fns[i % NCOPIES]()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters

dev = "cuda"; m = 1
mul1 = torch.tensor([MUL1_MARKER_SIGNED_INT32], dtype=torch.int32, device=dev)
tot_e = tot_b = 0.0
print(f"{'module':14s} {'exl3 µs':>8s} {'bf16 µs':>8s} {'exl3 GB/s':>9s} {'bf16 GB/s':>9s}")
for name, k_in, n_out, K, cnt in SHAPES:
    lins, wbfs = [], []
    for _ in range(NCOPIES):
        tr = torch.randint(-32768, 32767, (k_in//16, n_out//16, 16*K),
                           dtype=torch.int16, device=dev)
        suh = torch.randn(k_in, dtype=torch.float16, device=dev)
        svh = torch.randn(n_out, dtype=torch.float16, device=dev)
        lins.append(make_linear_exl3(tr, suh, svh, None, mul1, out_dtype=torch.float16))
        wbfs.append(torch.randn(n_out, k_in, dtype=torch.bfloat16, device=dev))
    x16 = torch.randn(m, k_in, dtype=torch.float16, device=dev)
    xbf = torch.randn(m, k_in, dtype=torch.bfloat16, device=dev)
    t_e = time_rot([(lambda l=l: l.forward(x16, {}, out_dtype=torch.float16)) for l in lins])
    t_b = time_rot([(lambda w=w: F.linear(xbf, w)) for w in wbfs])
    by_e = k_in*n_out*K/8; by_b = k_in*n_out*2
    print(f"{name} {t_e:8.1f} {t_b:8.1f} {by_e/t_e/1e3:9.1f} {by_b/t_b/1e3:9.1f}")
    tot_e += t_e*cnt; tot_b += t_b*cnt
print(f"TOTAL pondéré (hors q_a/kv_a/mlp.down): exl3 {tot_e/1000:.2f} ms vs bf16 {tot_b/1000:.2f} ms")
