"""
Script 4: FP8 Quality Regression Check
Compares TRT-LLM FP8 outputs against HF BF16 baseline using greedy decoding.
"""

import csv
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorrt_llm._tensorrt_engine import LLM as TRTEngineLLM
from tensorrt_llm.llmapi import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams as TRTSamplingParams

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
os.makedirs("results", exist_ok=True)

QUALITY_CHECKS = [
    ("What is the capital of France?", ["paris"]),
    ("Explain what a neural network is in one sentence.", ["layer", "neuron", "weight", "network", "learn"]),
    ("What is 17 multiplied by 13?", ["221"]),
    ("Name the three laws of thermodynamics.", ["energy", "entropy", "first", "second", "third"]),
    ("What programming language was created by Guido van Rossum?", ["python"]),
]

PROMPTS = [q[0] for q in QUALITY_CHECKS]


def check_factual(text: str, keywords: list[str]) -> bool:
    text = text.lower()
    matches = sum(kw.lower() in text for kw in keywords)
    return matches >= max(1, len(keywords) // 2)


print("Loading HF BF16 baseline...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

hf_outputs = []

for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = hf_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.0,
        )

    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    hf_outputs.append(text[:200])
    print(f"HF: {text[:80]}")

del hf_model
gc.collect()
torch.cuda.empty_cache()


print("\nLoading TRT-LLM FP8...")
trt_fp8 = TRTEngineLLM(
    model=MODEL_ID,
    dtype="bfloat16",
    quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
)

sampling_params = TRTSamplingParams(
    temperature=0.0,
    max_tokens=50,
)

fp8_outputs = []

for prompt in PROMPTS:
    output = trt_fp8.generate([prompt], sampling_params=sampling_params)
    text = output[0].outputs[0].text.strip()
    fp8_outputs.append(text[:200])
    print(f"FP8: {text[:80]}")


quality_results = []

for i, ((prompt, keywords), hf, fp8) in enumerate(
    zip(QUALITY_CHECKS, hf_outputs, fp8_outputs)
):
    factual_pass = check_factual(fp8, keywords)
    surface_match = hf[:40].lower().strip() == fp8[:40].lower().strip()

    quality_results.append({
        "prompt_id": i,
        "prompt": prompt,
        "required_keywords": "|".join(keywords),
        "hf_output": hf,
        "fp8_output": fp8,
        "surface_match": surface_match,
        "factual_pass": factual_pass,
    })

    print(f"\nQ{i+1}: {'PASS' if factual_pass else 'FAIL'}")
    print(f"HF : {hf[:100]}")
    print(f"FP8: {fp8[:100]}")


total = len(quality_results)
factual_passes = sum(r["factual_pass"] for r in quality_results)

print(f"\nFactual passes: {factual_passes}/{total}")
print(f"Overall verdict: {'PASS' if factual_passes == total else 'FAIL'}")

with open("results/quality_regression.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=quality_results[0].keys())
    writer.writeheader()
    writer.writerows(quality_results)

print("Saved results/quality_regression.csv")
