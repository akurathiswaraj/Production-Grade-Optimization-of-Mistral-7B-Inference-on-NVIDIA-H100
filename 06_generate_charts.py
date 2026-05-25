"""
Script 6: Generate Benchmark Charts
=====================================
Reads CSVs from scripts 01 and 02 only (HF baseline + vLLM).
TRT-LLM scripts 03/04 were skipped due to TRT-LLM 0.18.0 compatibility
issues with Mistral's embed_positions weight registration.
Saves to results/benchmark_charts.png
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

hf_toks    = load_col("results/hf_baseline.csv", "tok_per_sec")
hf_latency = load_col("results/hf_baseline.csv", "total_ms")
hf_vram    = load_col("results/hf_baseline.csv", "peak_mem_gb")

vllm_toks    = load_col("results/vllm_benchmark.csv", "tok_per_sec")
vllm_latency = load_col("results/vllm_benchmark.csv", "total_ms")

vllm_sweep = load_rows("results/vllm_batch_sweep.csv")
batch_sizes  = [int(r["batch_size"])            for r in vllm_sweep]
vllm_total   = [float(r["total_tok_per_sec"])   for r in vllm_sweep]
vllm_per_req = [float(r["per_req_tok_per_sec"]) for r in vllm_sweep]

# ── Aggregates ────────────────────────────────────────────────────────────────
stacks      = ["HF BF16 (Baseline)", "vLLM BF16"]
toks_p50    = [np.percentile(hf_toks, 50),    np.percentile(vllm_toks, 50)]
toks_p95    = [np.percentile(hf_toks, 95),    np.percentile(vllm_toks, 95)]
latency_p50 = [np.percentile(hf_latency, 50), np.percentile(vllm_latency, 50)]
latency_p95 = [np.percentile(hf_latency, 95), np.percentile(vllm_latency, 95)]
memory_gb   = [np.mean(hf_vram), 14.0]   # vLLM VRAM from runtime log

speedup = toks_p50[1] / toks_p50[0]
latency_reduction = (1 - latency_p50[1] / latency_p50[0]) * 100

print(f"\nThroughput p50  — HF: {toks_p50[0]:.1f}  vLLM: {toks_p50[1]:.1f}")
print(f"Latency p50 ms  — HF: {latency_p50[0]:.1f}  vLLM: {latency_p50[1]:.1f}")
print(f"Speedup: {speedup:.2f}x  |  Latency reduction: {latency_reduction:.0f}%")

# ── Style ─────────────────────────────────────────────────────────────────────
HF_COLOR   = "#a855f7"
VLLM_COLOR = "#00b4ff"
BG_DARK    = "#050810"
BG_AX      = "#0d1424"
GRID_COLOR = "#1e2d45"
TEXT_MUTED = "#94a3b8"

def style_ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.xaxis.label.set_color(TEXT_MUTED)

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG_DARK)

bar_colors = [HF_COLOR, VLLM_COLOR]

# ── Chart 1: Throughput p50 ───────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1); style_ax(ax1)
bars = ax1.bar(stacks, toks_p50, color=bar_colors, width=0.5, edgecolor="none")
for b, v in zip(bars, toks_p50):
    ax1.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}", ha="center",
             color="white", fontsize=11, fontweight="bold")
ax1.set_title("Throughput p50 (tok/s)", color="white", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Tokens / second", fontsize=10)
ax1.annotate(f"{speedup:.1f}x faster →", xy=(1, toks_p50[1]), xytext=(0.6, toks_p50[1]*0.6),
             color=VLLM_COLOR, fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=VLLM_COLOR))

# ── Chart 2: Latency p50 ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2); style_ax(ax2)
bars2 = ax2.bar(stacks, latency_p50, color=bar_colors, width=0.5, edgecolor="none")
for b, v in zip(bars2, latency_p50):
    ax2.text(b.get_x()+b.get_width()/2, v+8, f"{v:.0f}ms", ha="center",
             color="white", fontsize=11, fontweight="bold")
ax2.set_title("Latency p50 (ms, lower=better)", color="white", fontsize=12, fontweight="bold", pad=10)
ax2.set_ylabel("Milliseconds", fontsize=10)

# ── Chart 3: p50 vs p95 grouped ──────────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3); style_ax(ax3)
x = np.arange(2)
w = 0.3
b1 = ax3.bar(x - w/2, toks_p50, w, label="p50", color=bar_colors, edgecolor="none")
b2 = ax3.bar(x + w/2, toks_p95, w, label="p95", color=[HF_COLOR, VLLM_COLOR],
             edgecolor="none", alpha=0.55)
ax3.set_title("Throughput p50 vs p95 (tok/s)", color="white", fontsize=12, fontweight="bold", pad=10)
ax3.set_xticks(x); ax3.set_xticklabels(stacks, fontsize=9)
ax3.set_ylabel("Tokens / second", fontsize=10)
ax3.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COLOR, labelcolor="white")

# ── Chart 4: Peak GPU memory ──────────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4); style_ax(ax4)
bars4 = ax4.bar(stacks, memory_gb, color=bar_colors, width=0.5, edgecolor="none")
for b, v in zip(bars4, memory_gb):
    ax4.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f} GB", ha="center",
             color="white", fontsize=11, fontweight="bold")
ax4.set_title("Peak GPU Memory (lower=better)", color="white", fontsize=12, fontweight="bold", pad=10)
ax4.set_ylabel("Gigabytes", fontsize=10)
ax4.set_ylim(0, 18)

# ── Chart 5: vLLM total throughput batch sweep ────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5); style_ax(ax5)
ax5.plot(batch_sizes, vllm_total, "o-", color=VLLM_COLOR, linewidth=2.5,
         markersize=8, label="vLLM Total tok/s")
for x_, y_ in zip(batch_sizes, vllm_total):
    ax5.text(x_, y_+20, f"{y_:.0f}", ha="center", color=VLLM_COLOR, fontsize=9)
ax5.set_title("vLLM Total Throughput vs Batch Size", color="white", fontsize=12, fontweight="bold", pad=10)
ax5.set_xlabel("Batch Size", fontsize=10)
ax5.set_ylabel("Total tok/s", fontsize=10)
ax5.set_xticks(batch_sizes)
ax5.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COLOR, labelcolor="white")

# ── Chart 6: vLLM per-request throughput batch sweep ─────────────────────────
ax6 = fig.add_subplot(2, 3, 6); style_ax(ax6)
ax6.plot(batch_sizes, vllm_per_req, "s-", color="#00f5a0", linewidth=2.5,
         markersize=8, label="vLLM per-request tok/s")
for x_, y_ in zip(batch_sizes, vllm_per_req):
    ax6.text(x_, y_+1, f"{y_:.0f}", ha="center", color="#00f5a0", fontsize=9)
ax6.set_title("vLLM Per-Request Throughput vs Batch Size", color="white", fontsize=12, fontweight="bold", pad=10)
ax6.set_xlabel("Batch Size", fontsize=10)
ax6.set_ylabel("tok/s per request", fontsize=10)
ax6.set_xticks(batch_sizes)
ax6.legend(fontsize=9, facecolor=BG_AX, edgecolor=GRID_COLOR, labelcolor="white")

plt.suptitle(
    f"Mistral-7B H100: HF BF16 → vLLM  |  {speedup:.1f}x Throughput  ·  {latency_reduction:.0f}% Latency Reduction",
    color="white", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("results/benchmark_charts.png", dpi=150, bbox_inches="tight",
            facecolor=BG_DARK, edgecolor="none")
plt.show()
print(f"\nSaved results/benchmark_charts.png")
print(f"Speedup: {speedup:.2f}x  |  Latency reduction: {latency_reduction:.0f}%")
