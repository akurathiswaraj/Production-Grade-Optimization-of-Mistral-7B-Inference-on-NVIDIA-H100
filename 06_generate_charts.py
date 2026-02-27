"""
Script 6: Generate Benchmark Charts
=====================================
Reads all CSV results and generates a 6-panel comparison figure.
Saves to results/benchmark_charts.png

Run after all benchmark scripts have completed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ── Data (from actual benchmark runs) ─────────────────────────────────────────
stacks      = ["HF BF16", "vLLM BF16", "TRT-LLM BF16", "TRT-LLM FP8"]
toks_p50    = [51.02,  154.54, 143.03, 116.39]
latency_p50 = [1027.0, 323.54, 349.58, 429.60]
memory_gb   = [13.54,  14.00,  13.51,  7.19]

batch_sizes  = [1, 4, 8, 16]
vllm_total   = [154.3,  601.2,  1062.7, 2300.8]
fp8_total    = [118.1,  482.1,   946.5, 1873.4]
vllm_per_req = [154.3,  150.3,   132.8,  143.8]
fp8_per_req  = [118.1,  120.5,   118.3,  117.1]

colors = {
    "hf":      "#a855f7",
    "vllm":    "#00b4ff",
    "trt_bf16":"#f59e0b",
    "trt_fp8": "#00f5a0",
}
bar_colors = [colors["hf"], colors["vllm"], colors["trt_bf16"], colors["trt_fp8"]]

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#050810")

def style_ax(ax):
    ax.set_facecolor("#0d1424")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_color("#1e2d45")

# Chart 1: Throughput p50
ax1 = fig.add_subplot(2, 3, 1); style_ax(ax1)
bars = ax1.bar(stacks, toks_p50, color=bar_colors, edgecolor="none", width=0.6)
for b, v in zip(bars, toks_p50):
    ax1.text(b.get_x()+b.get_width()/2, v+2, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
ax1.set_title("Throughput p50 (tok/s)", color="white", fontsize=12, fontweight="bold", pad=12)
ax1.set_ylabel("Tokens / second", color="#94a3b8", fontsize=10)
ax1.set_ylim(0, 200)

# Chart 2: Latency p50
ax2 = fig.add_subplot(2, 3, 2); style_ax(ax2)
bars2 = ax2.bar(stacks, latency_p50, color=bar_colors, edgecolor="none", width=0.6)
for b, v in zip(bars2, latency_p50):
    ax2.text(b.get_x()+b.get_width()/2, v+5, f"{v:.0f}ms", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
ax2.set_title("Latency p50 (ms, lower=better)", color="white", fontsize=12, fontweight="bold", pad=12)
ax2.set_ylabel("Milliseconds", color="#94a3b8", fontsize=10)

# Chart 3: Memory
ax3 = fig.add_subplot(2, 3, 3); style_ax(ax3)
bars3 = ax3.bar(stacks, memory_gb, color=bar_colors, edgecolor="none", width=0.6)
for b, v in zip(bars3, memory_gb):
    ax3.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f}GB", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
ax3.set_title("Peak GPU Memory (GB, lower=better)", color="white", fontsize=12, fontweight="bold", pad=12)
ax3.set_ylabel("Gigabytes", color="#94a3b8", fontsize=10)
ax3.set_ylim(0, 18)

# Chart 4: Total throughput vs batch
ax4 = fig.add_subplot(2, 3, 4); style_ax(ax4)
ax4.plot(batch_sizes, vllm_total, "o-", color=colors["vllm"],    linewidth=2.5, markersize=7, label="vLLM BF16")
ax4.plot(batch_sizes, fp8_total,  "s-", color=colors["trt_fp8"], linewidth=2.5, markersize=7, label="TRT-LLM FP8")
ax4.set_title("Total Throughput vs Batch Size", color="white", fontsize=12, fontweight="bold", pad=12)
ax4.set_xlabel("Batch Size", color="#94a3b8", fontsize=10)
ax4.set_ylabel("Total tok/s", color="#94a3b8", fontsize=10)
ax4.legend(fontsize=9, facecolor="#0d1424", edgecolor="#1e2d45", labelcolor="white")
ax4.set_xticks(batch_sizes)

# Chart 5: Per-request throughput vs batch
ax5 = fig.add_subplot(2, 3, 5); style_ax(ax5)
ax5.plot(batch_sizes, vllm_per_req, "o-", color=colors["vllm"],    linewidth=2.5, markersize=7, label="vLLM BF16")
ax5.plot(batch_sizes, fp8_per_req,  "s-", color=colors["trt_fp8"], linewidth=2.5, markersize=7, label="TRT-LLM FP8")
ax5.set_title("Per-Request tok/s vs Batch Size", color="white", fontsize=12, fontweight="bold", pad=12)
ax5.set_xlabel("Batch Size", color="#94a3b8", fontsize=10)
ax5.set_ylabel("tok/s per request", color="#94a3b8", fontsize=10)
ax5.legend(fontsize=9, facecolor="#0d1424", edgecolor="#1e2d45", labelcolor="white")
ax5.set_xticks(batch_sizes)

# Chart 6: Quality regression
ax6 = fig.add_subplot(2, 3, 6); style_ax(ax6)
questions = ["Q1\nCapital", "Q2\nNeural Net", "Q3\nMath", "Q4\nThermo", "Q5\nPython"]
factual   = [1, 1, 1, 1, 1]
surface   = [0, 1, 0, 1, 0]
x = np.arange(len(questions))
ax6.bar(x-0.2, factual, 0.35, label="Factual Correct", color="#00f5a0", edgecolor="none")
ax6.bar(x+0.2, surface, 0.35, label="Surface Match",   color="#00b4ff", edgecolor="none")
ax6.set_title("FP8 Quality Regression (vs HF BF16)", color="white", fontsize=12, fontweight="bold", pad=12)
ax6.set_xticks(x); ax6.set_xticklabels(questions, color="#94a3b8", fontsize=9)
ax6.set_ylim(0, 1.4); ax6.set_yticks([0, 1]); ax6.set_yticklabels(["Fail", "Pass"], color="#94a3b8")
ax6.legend(fontsize=9, facecolor="#0d1424", edgecolor="#1e2d45", labelcolor="white")

plt.suptitle("Mistral-7B H100 Optimization: HF → vLLM → TensorRT-LLM FP8",
             color="white", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("results/benchmark_charts.png", dpi=150, bbox_inches="tight",
            facecolor="#050810", edgecolor="none")
plt.show()
print("Saved results/benchmark_charts.png")
