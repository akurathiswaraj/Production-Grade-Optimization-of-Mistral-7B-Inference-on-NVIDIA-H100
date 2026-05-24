"""
Script 6: Generate Benchmark Charts
Reads CSV result files and generates benchmark charts.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)


def load_col(path, col):
    with open(path, newline="") as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


print("Reading result CSVs...")

hf_toks = load_col("results/hf_baseline.csv", "tok_per_sec")
hf_latency = load_col("results/hf_baseline.csv", "total_ms")
hf_vram = load_col("results/hf_baseline.csv", "peak_mem_gb")

vllm_toks = load_col("results/vllm_benchmark.csv", "tok_per_sec")
vllm_latency = load_col("results/vllm_benchmark.csv", "total_ms")

trt_bf16_toks = load_col("results/trt_bf16_benchmark.csv", "tok_per_sec")
trt_bf16_latency = load_col("results/trt_bf16_benchmark.csv", "total_ms")

trt_fp8_toks = load_col("results/trt_fp8_benchmark.csv", "tok_per_sec")
trt_fp8_latency = load_col("results/trt_fp8_benchmark.csv", "total_ms")

vllm_sweep = load_rows("results/vllm_batch_sweep.csv")
fp8_sweep = load_rows("results/trt_fp8_batch_sweep.csv")

batch_sizes = [int(r["batch_size"]) for r in vllm_sweep]
vllm_total = [float(r["total_tok_per_sec"]) for r in vllm_sweep]
fp8_total = [float(r["total_tok_per_sec"]) for r in fp8_sweep]
vllm_per_req = [float(r["per_req_tok_per_sec"]) for r in vllm_sweep]
fp8_per_req = [float(r["per_req_tok_per_sec"]) for r in fp8_sweep]

qual_rows = load_rows("results/quality_regression.csv")
factual_vals = [1 if r["factual_pass"] == "True" else 0 for r in qual_rows]
surface_vals = [1 if r["surface_match"] == "True" else 0 for r in qual_rows]

stacks = ["HF BF16", "vLLM BF16", "TRT-LLM BF16", "TRT-LLM FP8"]

toks_p50 = [
    np.percentile(hf_toks, 50),
    np.percentile(vllm_toks, 50),
    np.percentile(trt_bf16_toks, 50),
    np.percentile(trt_fp8_toks, 50),
]

latency_p50 = [
    np.percentile(hf_latency, 50),
    np.percentile(vllm_latency, 50),
    np.percentile(trt_bf16_latency, 50),
    np.percentile(trt_fp8_latency, 50),
]

# vLLM/TRT VRAM values collected from runtime logs because these CSVs do not store VRAM.
memory_gb = [np.mean(hf_vram), 14.0, 13.51, 7.19]

print("Throughput p50:", toks_p50)
print("Latency p50:", latency_p50)
print("Memory GB:", memory_gb)

colors = {
    "hf": "#a855f7",
    "vllm": "#00b4ff",
    "trt_bf16": "#f59e0b",
    "trt_fp8": "#00f5a0",
}

bar_colors = [
    colors["hf"],
    colors["vllm"],
    colors["trt_bf16"],
    colors["trt_fp8"],
]

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#050810")


def style_ax(ax):
    ax.set_facecolor("#0d1424")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e2d45")


ax1 = fig.add_subplot(2, 3, 1)
style_ax(ax1)
bars = ax1.bar(stacks, toks_p50, color=bar_colors)
for b, v in zip(bars, toks_p50):
    ax1.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", color="white")
ax1.set_title("Throughput p50 (tok/s)", color="white")
ax1.set_ylabel("Tokens/sec", color="#94a3b8")

ax2 = fig.add_subplot(2, 3, 2)
style_ax(ax2)
bars = ax2.bar(stacks, latency_p50, color=bar_colors)
for b, v in zip(bars, latency_p50):
    ax2.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f}ms", ha="center", color="white")
ax2.set_title("Latency p50 (ms)", color="white")
ax2.set_ylabel("Milliseconds", color="#94a3b8")

ax3 = fig.add_subplot(2, 3, 3)
style_ax(ax3)
bars = ax3.bar(stacks, memory_gb, color=bar_colors)
for b, v in zip(bars, memory_gb):
    ax3.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}GB", ha="center", color="white")
ax3.set_title("Peak GPU Memory", color="white")
ax3.set_ylabel("GB", color="#94a3b8")

ax4 = fig.add_subplot(2, 3, 4)
style_ax(ax4)
ax4.plot(batch_sizes, vllm_total, "o-", color=colors["vllm"], label="vLLM BF16")
ax4.plot(batch_sizes, fp8_total, "s-", color=colors["trt_fp8"], label="TRT-LLM FP8")
ax4.set_title("Total Throughput vs Batch Size", color="white")
ax4.set_xlabel("Batch Size", color="#94a3b8")
ax4.set_ylabel("Total tok/s", color="#94a3b8")
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
style_ax(ax5)
ax5.plot(batch_sizes, vllm_per_req, "o-", color=colors["vllm"], label="vLLM BF16")
ax5.plot(batch_sizes, fp8_per_req, "s-", color=colors["trt_fp8"], label="TRT-LLM FP8")
ax5.set_title("Per-Request Throughput vs Batch Size", color="white")
ax5.set_xlabel("Batch Size", color="#94a3b8")
ax5.set_ylabel("tok/s per request", color="#94a3b8")
ax5.legend()

ax6 = fig.add_subplot(2, 3, 6)
style_ax(ax6)
questions = [f"Q{i+1}" for i in range(len(qual_rows))]
x = np.arange(len(questions))
ax6.bar(x - 0.2, factual_vals, 0.35, label="Factual Pass", color=colors["trt_fp8"])
ax6.bar(x + 0.2, surface_vals, 0.35, label="Surface Match", color=colors["vllm"])
ax6.set_title("FP8 Quality Regression", color="white")
ax6.set_xticks(x)
ax6.set_xticklabels(questions)
ax6.set_ylim(0, 1.4)
ax6.set_yticks([0, 1])
ax6.set_yticklabels(["Fail", "Pass"])
ax6.legend()

plt.suptitle(
    "Mistral-7B H100 Optimization: HF → vLLM → TensorRT-LLM FP8",
    color="white",
    fontsize=15,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(
    "results/benchmark_charts.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#050810",
)

plt.show()
print("Saved results/benchmark_charts.png")
