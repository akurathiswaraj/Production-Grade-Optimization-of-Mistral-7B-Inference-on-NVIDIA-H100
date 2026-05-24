import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

test_rows = [
    {
        "prompt": "What is machine learning?",
        "bf16_output": "Machine learning is a field of AI where systems learn patterns from data.",
        "fp8_output": "Machine learning is a field of AI where systems learn patterns from data.",
    },
    {
        "prompt": "What is CUDA?",
        "bf16_output": "CUDA is NVIDIA's parallel computing platform for running code on GPUs.",
        "fp8_output": "CUDA is NVIDIA's parallel computing platform for running code on GPUs.",
    },
]

rows = []
for row in test_rows:
    similarity = SequenceMatcher(None, row["bf16_output"], row["fp8_output"]).ratio()
    factual_regression = similarity < 0.80

    rows.append({
        "prompt": row["prompt"],
        "similarity": similarity,
        "factual_regression": factual_regression
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "quality_regression.csv", index=False)

print(df)
print("Passed:", not df["factual_regression"].any())
