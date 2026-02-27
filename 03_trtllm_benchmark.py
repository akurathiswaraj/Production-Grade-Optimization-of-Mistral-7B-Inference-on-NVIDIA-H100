"""
Script 3: TensorRT-LLM BF16 + FP8 Benchmark
=============================================
Benchmarks Mistral-7B-Instruct-v0.3 using TensorRT-LLM 1.0.
Runs BF16 (PyTorch backend) then FP8 (TensorRT engine backend).
Saves results to results/trt_bf16_benchmark.csv, trt_fp8_benchmark.csv,
               results/trt_fp8_batch_sweep.csv

Install: pip install tensorrt-llm onnx==1.16.0
         apt-get install -y libopenmpi-dev openmpi-bin
"""

import csv
import os
import time
import gc
import numpy as np
import torch

from tensorrt_llm import LLM as TRTLLM
from tensorrt_llm.sampling_params import SamplingParams as TRTSamplingParams
from tensorrt_llm._tensorrt_engine import LLM as TRTEngineLLM
from tensorrt_llm.llmapi import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS  = 50
os.makedirs("results", exist_ok=True)

PROMPTS = [
    "Explain KV cache in transformers in 2 sentences.",
    "What is the difference between attention and self-attention?",
    "Describe how gradient descent works.",
    "What is layer normalization and why is it used?",
    "Explain positional encoding in transformers.",
    "What is the softmax function used for in attention?",
    "How does dropout prevent overfitting?",
    "What is the purpose of the feed-forward layer in transformers?",
    "Explain the difference between encoder and decoder stacks.",
    "What is tokenization and why does it matter?",
]

sampling_params = TRTSamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

def run_benchmark(engine, label):
    results = []
    print(f"\nRunning {label} benchmark...")
    for i, prompt in enumerate(PROMPTS):
        start   = time.perf_counter()
        outputs = engine.generate([prompt], sampling_params=sampling_params)
        end     = time.perf_counter()

        total_ms         = (end - start) * 1000
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        tok_per_sec      = (tokens_generated / total_ms) * 1000

        results.append({
            "prompt_id":        i,
            "total_ms":         total_ms,
            "tokens_generated": tokens_generated,
            "tok_per_sec":      tok_per_sec,
        })
        print(f"  [{i+1}/{len(PROMPTS)}] total: {total_ms:.1f}ms | tok/s: {tok_per_sec:.1f}")

    toks  = [r["tok_per_sec"] for r in results]
    times = [r["total_ms"]    for r in results]
    print(f"\n--- {label} RESULTS ---")
    print(f"Latency p50: {np.percentile(times, 50):.2f} ms")
    print(f"Latency p95: {np.percentile(times, 95):.2f} ms")
    print(f"Tok/s p50:   {np.percentile(toks,  50):.2f}")
    print(f"Tok/s p95:   {np.percentile(toks,  95):.2f}")
    return results

# ── TRT-LLM BF16 ─────────────────────────────────────────────────────────────
print("Loading TRT-LLM BF16 (PyTorch backend)...")
trt_bf16 = TRTLLM(model=MODEL_ID, dtype="bfloat16")
results_bf16 = run_benchmark(trt_bf16, "TRT-LLM BF16")

with open("results/trt_bf16_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results_bf16[0].keys())
    writer.writeheader()
    writer.writerows(results_bf16)
print("Saved results/trt_bf16_benchmark.csv")

del trt_bf16
gc.collect()
torch.cuda.empty_cache()

# ── TRT-LLM FP8 ──────────────────────────────────────────────────────────────
print("\nBuilding TRT-LLM FP8 engine (~3 mins)...")
trt_fp8 = TRTEngineLLM(
    model=MODEL_ID,
    dtype="bfloat16",
    quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
)
results_fp8 = run_benchmark(trt_fp8, "TRT-LLM FP8")

with open("results/trt_fp8_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results_fp8[0].keys())
    writer.writeheader()
    writer.writerows(results_fp8)
print("Saved results/trt_fp8_benchmark.csv")

# ── FP8 Batch sweep ───────────────────────────────────────────────────────────
print("\nRunning FP8 batch sweep...")
batch_results = []

for bs in [1, 4, 8, 16]:
    batch_prompts = (PROMPTS * 10)[:bs]
    start   = time.perf_counter()
    outputs = trt_fp8.generate(batch_prompts, sampling_params=sampling_params)
    end     = time.perf_counter()

    total_ms     = (end - start) * 1000
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec  = (total_tokens / total_ms) * 1000

    batch_results.append({
        "batch_size":          bs,
        "total_tok_per_sec":   tok_per_sec,
        "per_req_tok_per_sec": tok_per_sec / bs,
        "latency_ms":          total_ms,
    })
    print(f"  Batch {bs:>2}: {total_ms:.1f}ms | {tok_per_sec:.1f} tok/s | {tok_per_sec/bs:.1f} per req")

with open("results/trt_fp8_batch_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=batch_results[0].keys())
    writer.writeheader()
    writer.writerows(batch_results)
print("Saved results/trt_fp8_batch_sweep.csv")
