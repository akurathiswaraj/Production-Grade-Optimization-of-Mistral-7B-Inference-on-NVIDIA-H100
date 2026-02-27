# Mistral-7B H100 Optimization: HF → vLLM → TensorRT-LLM FP8

Production-grade benchmarking of Mistral-7B-Instruct-v0.3 across three inference stacks on NVIDIA H100 SXM5.

![Benchmark Charts](results/benchmark_charts.png)

---

## Results Summary

| Stack | Tok/s p50 | Latency p50 | Peak VRAM | vs Baseline |
|---|---|---|---|---|
| HuggingFace PyTorch BF16 | 51.02 | 1027ms | 13.54 GB | 1x |
| vLLM BF16 | 154.54 | 323ms | ~14 GB | **3.03x** |
| TRT-LLM BF16 | 143.03 | 349ms | 13.51 GB | 2.8x |
| TRT-LLM FP8 | 116.39 | 429ms | **7.19 GB** | 2.3x |

### Key Findings

1. **Framework selection > hardware.** vLLM achieved 3x improvement over HF baseline on identical hardware through PagedAttention + continuous batching — no model changes required.

2. **FP8 is a memory optimization at low batch sizes.** At batch=1–16, FP8 provides no throughput advantage but halves model memory (13.5GB → 7.2GB), enabling 2x model density per GPU in production.

3. **vLLM scales near-linearly with batch size.** Total throughput went from 154 → 2300 tok/s (15x) from batch=1 to batch=16 with <10% latency increase.

4. **HF decode is CPU-bound, not GPU-bound.** PyTorch profiler revealed a 7:1 CPU/CUDA time ratio during decode — the H100 was idle 85% of the time waiting for Python kernel dispatch. This explains the 3x vLLM speedup.

5. **FP8 passes quality gate.** Zero factual regressions across 5 test prompts versus HF BF16 baseline.

### Batch Sweep

| Batch | vLLM tok/s | TRT-LLM FP8 tok/s |
|---|---|---|
| 1 | 154 | 118 |
| 4 | 601 | 482 |
| 8 | 1062 | 946 |
| 16 | 2300 | 1873 |

---

## Hardware & Environment

| Component | Spec |
|---|---|
| GPU | NVIDIA H100 SXM5 80GB HBM3 |
| Cloud | Modal Labs ($4.74/hr) |
| CUDA | 12.1+ |
| Python | 3.12 |
| PyTorch | 2.7.1 |
| vLLM | 0.16.0 |
| TensorRT-LLM | 1.0.0 |

**Estimated cost to reproduce:** ~$15–20

---

## Repository Structure

```
mistral-h100-optimization/
├── scripts/
│   ├── 01_hf_baseline.py          # HuggingFace PyTorch BF16 benchmark
│   ├── 02_vllm_benchmark.py       # vLLM benchmark + batch sweep
│   ├── 03_trtllm_benchmark.py     # TRT-LLM BF16 + FP8 benchmark
│   ├── 04_quality_regression.py   # FP8 quality validation
│   ├── 05_profiler.py             # PyTorch profiler (prefill + decode cases)
│   └── 06_generate_charts.py      # Reproduce benchmark charts
├── results/
│   ├── hf_baseline.csv
│   ├── vllm_benchmark.csv
│   ├── vllm_batch_sweep.csv
│   ├── trt_bf16_benchmark.csv
│   ├── trt_fp8_benchmark.csv
│   ├── trt_fp8_batch_sweep.csv
│   ├── quality_regression.csv
│   ├── profiling_summary.md
│   └── benchmark_charts.png
└── README.md
```

---

## Reproducing Results

### 1. Provision Hardware

Rent an H100 SXM5 (80GB) instance. Recommended providers:
- [Modal Labs](https://modal.com) — notebook environment used in this project
- [Lambda Labs](https://lambdalabs.com)
- [RunPod](https://runpod.io)

### 2. Install Dependencies

```bash
# HuggingFace baseline
pip install transformers torch accelerate numpy

# vLLM
pip install vllm

# TensorRT-LLM
apt-get install -y libopenmpi-dev openmpi-bin
pip install tensorrt-llm onnx==1.16.0
```

### 3. Run Scripts in Order

```bash
# Run each script sequentially (each frees GPU memory before the next)
python scripts/01_hf_baseline.py
python scripts/02_vllm_benchmark.py
python scripts/03_trtllm_benchmark.py
python scripts/04_quality_regression.py
python scripts/05_profiler.py
python scripts/06_generate_charts.py
```

> **Note:** Each script is independent. They free GPU memory at the end so you can run them in sequence in a single session.

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
- **Prompts:** 10 fixed prompts, repeated consistently across all stacks
- **Decoding:** Greedy (temperature=0, fixed seed) for reproducibility
- **Output length:** 50 tokens (`max_new_tokens=50`)
- **Warmup:** 3 passes before each benchmark to eliminate JIT compilation noise
- **Timing:** CUDA events (`torch.cuda.Event`) for GPU-accurate measurement on HF; wall-clock `perf_counter` for vLLM and TRT-LLM server stacks
- **Statistics:** p50/p95 computed over 10 prompt runs

---

## Profiling Summary

### Case 1: Prefill-Heavy (1500 token prompt, 32 output tokens)
- Total CUDA time: **260ms**
- Top kernel: `aten::mm` (71.22% of CUDA time)
- CPU/CUDA ratio: **7.1x** — Python overhead dominates even in prefill

### Case 2: Decode-Heavy (short prompt, 256 output tokens)
- Total CUDA time: **2006ms**
- Top kernel: `aten::mm` (72.49% of CUDA time, 57,600 calls)
- CPU/CUDA ratio: **7.1x** — H100 idle 85% of decode time

The consistent 7:1 CPU/CUDA ratio across both cases confirms that HuggingFace Transformers is CPU-bound due to Python's per-token kernel dispatch overhead. vLLM's C++ continuous batching scheduler eliminates this bottleneck.

---

## Production Recommendation

| Use Case | Recommended Stack | Reason |
|---|---|---|
| Latency-sensitive, single request | vLLM BF16 | 324ms p50, 154 tok/s |
| Memory-constrained deployment | TRT-LLM FP8 | 7.2GB — fit 2 models per GPU |
| Maximum throughput at scale | vLLM batch≥16 | 2300 tok/s at batch=16 |
