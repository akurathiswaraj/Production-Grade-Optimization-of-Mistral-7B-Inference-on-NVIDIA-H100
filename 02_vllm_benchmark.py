"""
Script 2: vLLM BF16 Benchmark
==============================
Benchmarks Mistral-7B-Instruct-v0.3 using vLLM with PagedAttention.
Runs single-request benchmark + batch size sweep (1, 4, 8, 16).
Saves results to results/vllm_benchmark.csv and results/vllm_batch_sweep.csv

Install: pip install vllm
"""

import csv
import os
import time
import numpy as np
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 50
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

# ── Setup ─────────────────────────────────────────────────────────────────────
llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

# ── Single-request benchmark ──────────────────────────────────────────────────
results = []
print("Running vLLM single-request benchmark...")

for i, prompt in enumerate(PROMPTS):
    start   = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
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

print(f"\n--- vLLM RESULTS ---")
print(f"Total latency p50: {np.percentile(times, 50):.2f} ms")
print(f"Total latency p95: {np.percentile(times, 95):.2f} ms")
print(f"Tok/s p50:         {np.percentile(toks,  50):.2f}")
print(f"Tok/s p95:         {np.percentile(toks,  95):.2f}")

# ── Batch sweep ───────────────────────────────────────────────────────────────
print("\nRunning batch size sweep...")
batch_results = []

for bs in [1, 4, 8, 16]:
    batch_prompts = (PROMPTS * 10)[:bs]
    start   = time.perf_counter()
    outputs = llm.generate(batch_prompts, sampling_params)
    end     = time.perf_counter()

    total_ms     = (end - start) * 1000
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec  = (total_tokens / total_ms) * 1000

    batch_results.append({
        "batch_size":           bs,
        "total_tok_per_sec":    tok_per_sec,
        "per_req_tok_per_sec":  tok_per_sec / bs,
        "latency_ms":           total_ms,
    })
    print(f"  Batch {bs:>2}: {total_ms:.1f}ms | {tok_per_sec:.1f} tok/s total | {tok_per_sec/bs:.1f} tok/s per req")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("results/vllm_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with open("results/vllm_batch_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=batch_results[0].keys())
    writer.writeheader()
    writer.writerows(batch_results)

print("\nSaved results/vllm_benchmark.csv")
print("Saved results/vllm_batch_sweep.csv")
