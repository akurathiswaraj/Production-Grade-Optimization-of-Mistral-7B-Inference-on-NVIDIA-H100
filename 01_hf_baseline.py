"""
Script 1: HuggingFace PyTorch BF16 Baseline
============================================
Benchmarks Mistral-7B-Instruct-v0.3 using vanilla HuggingFace Transformers.
Measures TTFT, decode latency, tok/s, p50/p95, and peak VRAM.
Saves results to results/hf_baseline.csv
"""

import csv
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 50
WARMUP_RUNS    = 3
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
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda").eval()
print("Model loaded.")

# ── Warmup ────────────────────────────────────────────────────────────────────
print(f"Running {WARMUP_RUNS} warmup passes...")
warmup_inputs = tokenizer("Hello.", return_tensors="pt").to("cuda")
for _ in range(WARMUP_RUNS):
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()
print("Warmup done.")

# ── Benchmark ─────────────────────────────────────────────────────────────────
results = []
print("Running benchmark loop...")

for i, prompt in enumerate(PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()

    start_event       = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event         = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    first_token_event.record()

    with torch.no_grad():
        full_output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    end_event.record()
    torch.cuda.synchronize()

    ttft_ms          = start_event.elapsed_time(first_token_event)
    total_ms         = start_event.elapsed_time(end_event)
    decode_ms        = total_ms - ttft_ms
    tokens_generated = full_output.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec      = (tokens_generated / decode_ms) * 1000
    peak_mem_gb      = torch.cuda.max_memory_allocated() / 1024**3

    results.append({
        "prompt_id":        i,
        "ttft_ms":          ttft_ms,
        "decode_ms":        decode_ms,
        "total_ms":         total_ms,
        "tokens_generated": tokens_generated,
        "tok_per_sec":      tok_per_sec,
        "peak_mem_gb":      peak_mem_gb,
    })
    print(f"  [{i+1}/{len(PROMPTS)}] TTFT: {ttft_ms:.1f}ms | tok/s: {tok_per_sec:.1f}")

# ── Summary ───────────────────────────────────────────────────────────────────
ttfts = [r["ttft_ms"]    for r in results]
toks  = [r["tok_per_sec"] for r in results]

print(f"\n--- HF BASELINE RESULTS ---")
print(f"TTFT   p50: {np.percentile(ttfts, 50):.2f} ms")
print(f"TTFT   p95: {np.percentile(ttfts, 95):.2f} ms")
print(f"Tok/s  p50: {np.percentile(toks,  50):.2f}")
print(f"Tok/s  p95: {np.percentile(toks,  95):.2f}")
print(f"Avg VRAM:   {np.mean([r['peak_mem_gb'] for r in results]):.2f} GB")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("results/hf_baseline.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("Saved results/hf_baseline.csv")
