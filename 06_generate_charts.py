import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")

files = {
    "HF BF16": "hf_baseline.csv",
    "vLLM BF16": "vllm_benchmark.csv",
    "TRT BF16": "trt_bf16_benchmark.csv",
    "TRT FP8": "trt_fp8_benchmark.csv",
}

rows = []

for name, file in files.items():
    path = RESULTS_DIR / file
    if not path.exists():
        print(f"Missing: {path}")
        continue

    df = pd.read_csv(path)

    tok_col = next((c for c in df.columns if "tok" in c.lower()), None)
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    vram_col = next((c for c in df.columns if "vram" in c.lower() or "memory" in c.lower()), None)

    rows.append({
        "stack": name,
        "tokens_per_sec": df[tok_col].median() if tok_col else None,
        "latency_ms": df[lat_col].median() if lat_col else None,
        "vram_gb": df[vram_col].median() if vram_col else None,
    })

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(summary["stack"], summary["tokens_per_sec"])
plt.ylabel("Tokens/sec")
plt.title("Mistral-7B Throughput Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "benchmark_charts.png", dpi=200)

print(summary)
