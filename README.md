# Mistral-7B H100 Optimization: HF BF16 → vLLM

Benchmarking Mistral-7B-Instruct-v0.3 across two inference stacks on NVIDIA H100 80GB HBM3, with PyTorch profiler analysis identifying the root cause of HuggingFace's performance ceiling.

---

## Results Summary

| Stack | Tok/s p50 | Latency p50 | Peak VRAM | vs Baseline |
|---|---|---|---|---|
| HuggingFace PyTorch BF16 | 69.95 | 731ms | 13.54 GB | 1x |
| vLLM BF16 | 139.61 | 358ms | ~14 GB | **2.0x** |

### Key Findings

1. **vLLM achieves 2x throughput over HF baseline on identical hardware** (70 → 140 tok/s) through PagedAttention and continuous batching — no model changes required.

2. **HF decode is CPU-bound, not GPU-bound.** PyTorch profiler revealed a 3.8:1 CPU/CUDA time ratio during decode — the H100 was idle ~74% of decode time waiting for Python kernel dispatch. This is the root cause vLLM's C++ scheduler eliminates.

3. **The bottleneck is systemic, not prompt-dependent.** Both profiling cases (prefill-heavy and decode-heavy) showed the same 3.8:1 CPU/CUDA ratio, confirming Python overhead is architectural, not workload-specific.

4. **vLLM scales near-linearly with batch size.** Total throughput went from 140 → 2117 tok/s (15x) from batch=1 to batch=16 with <6% per-request latency increase.

5. **TRT-LLM skipped due to environment incompatibility.** TRT-LLM 0.18.0 has a known bug with Mistral's `embed_positions` weight registration that affects both the TRT engine builder and the pytorch backend. Scripts 03/04 are included but require TRT-LLM ≤0.14.0 to run.

### Profiler Details

| Case | CPU Total | CUDA Total | CPU/CUDA Ratio | H100 Idle |
|---|---|---|---|---|
| Prefill-heavy (1500-tok prompt, 32 out) | 1.037s | 0.284s | 3.66x | ~73% |
| Decode-heavy (short prompt, 256 out) | 8.033s | 2.119s | 3.79x | ~74% |

Top kernel both cases: `aten::mm` (~71% of CUDA time)

### Batch Sweep

| Batch | vLLM Total tok/s | Per-Request tok/s |
|---|---|---|
| 1 | 139.6 | 139.6 |
| 4 | 539.6 | 134.9 |
| 8 | 1061.1 | 132.6 |
| 16 | 2117.4 | 132.3 |

---

## Hardware & Environment

| Component | Spec |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| Cloud | Modal Labs |
| Python | 3.12.6 |
| PyTorch | 2.7.1 |
| vLLM | 0.8.5.post1 |
| TensorRT-LLM | 0.18.0 (incompatible — see Finding 5) |

---

## Repository Structure

```
├── 01_hf_baseline.py          # HuggingFace PyTorch BF16 benchmark
├── 02_vllm_benchmark.py       # vLLM benchmark + batch sweep
├── 03_trtllm_benchmark.py     # TRT-LLM (requires ≤0.14.0 — skipped)
├── 04_quality_regression.py   # FP8 regression check (requires ≤0.14.0 — skipped)
├── 05_profiler.py             # PyTorch profiler — prefill + decode cases
├── 06_generate_charts.py      # Reads CSVs, generates benchmark_charts.png
└── results/
    ├── hf_baseline.csv
    ├── vllm_benchmark.csv
    ├── vllm_batch_sweep.csv
    ├── trace_prefill_heavy.json
    ├── trace_decode_heavy.json
```

---

## Reproducing Results

### 1. Provision Hardware

Rent an H100 80GB instance:
- [Modal Labs](https://modal.com)
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)

### 2. Install Dependencies

```bash
pip install transformers torch accelerate numpy matplotlib
pip install vllm==0.8.5.post1
```

### 3. Run Scripts in Order

```bash
python 01_hf_baseline.py        # ~5 min  → hf_baseline.csv
python 02_vllm_benchmark.py     # ~5 min  → vllm_benchmark.csv, vllm_batch_sweep.csv
python 05_profiler.py           # ~10 min → Chrome trace JSON files
python 06_generate_charts.py    # ~1 min  → benchmark_charts.png
```

> Scripts 03 and 04 require TRT-LLM ≤0.14.0. Skip them if using 0.18.0+.

### 4. View Profiler Traces

```
# Open Chrome and navigate to:
chrome://tracing

# Load either trace file:
results/trace_prefill_heavy.json
results/trace_decode_heavy.json
```

---

## Benchmark Methodology

- **Model:** `mistralai/Mistral-7B-Instruct-v0.3` (same weights across all stacks)
- **Prompts:** 10 fixed prompts, repeated consistently across stacks
- **Decoding:** Greedy (temperature=0) for reproducibility
- **Output length:** 50 tokens (`max_new_tokens=50`)
- **Warmup:** 3 passes before each benchmark to eliminate JIT compilation noise
- **Timing:** CUDA Events (`torch.cuda.Event`) for GPU-accurate measurement on HF; wall-clock `perf_counter` for vLLM
- **Statistics:** p50/p95 computed over 10 prompt runs

---

## Profiling Summary

### Case 1: Prefill-Heavy (1500-token prompt, 32 output tokens)
- CPU total: **1.037s** | CUDA total: **283.6ms** | Ratio: **3.66x**
- Top kernel: `aten::mm` (70.11% of CUDA time)

### Case 2: Decode-Heavy (short prompt, 256 output tokens)
- CPU total: **8.033s** | CUDA total: **2.119s** | Ratio: **3.79x**
- Top kernel: `aten::mm` (71.53% of CUDA time, 57,600 calls)

The consistent 3.8x CPU/CUDA ratio across both cases confirms HuggingFace Transformers is CPU-bound due to Python's per-token kernel dispatch overhead — the H100 is idle ~74% of decode time. vLLM's C++ continuous batching scheduler eliminates this bottleneck, delivering the 2x throughput gain.

---

## Production Recommendation

| Use Case | Recommended Stack | Reason |
|---|---|---|
| Latency-sensitive, single request | vLLM BF16 | 358ms p50, 140 tok/s |
| Maximum throughput at scale | vLLM batch≥16 | 2117 tok/s at batch=16 |
