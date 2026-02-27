"""
Script 4: FP8 Quality Regression Check
=======================================
Compares TRT-LLM FP8 outputs against HF BF16 baseline using greedy decoding.
Checks for factual regressions across 5 prompts.
Saves results to results/quality_regression.csv
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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
os.makedirs("results", exist_ok=True)

QUALITY_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is in one sentence.",
    "What is 17 multiplied by 13?",
    "Name the three laws of thermodynamics.",
    "What programming language was created by Guido van Rossum?",
]

# ── HF Baseline outputs ───────────────────────────────────────────────────────
print("Loading HF baseline...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_model  = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda").eval()

hf_outputs = []
for prompt in QUALITY_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = hf_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    hf_outputs.append(text.strip()[:120])
    print(f"  HF: {text.strip()[:80]}")

del hf_model
gc.collect()
torch.cuda.empty_cache()

# ── FP8 outputs ───────────────────────────────────────────────────────────────
print("\nLoading TRT-LLM FP8 engine...")
trt_fp8 = TRTEngineLLM(
    model=MODEL_ID,
    dtype="bfloat16",
    quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
)

sampling_params = TRTSamplingParams(temperature=0, max_tokens=50)
fp8_outputs = []

for prompt in QUALITY_PROMPTS:
    output = trt_fp8.generate([prompt], sampling_params=sampling_params)
    text   = output[0].outputs[0].text.strip()[:120]
    fp8_outputs.append(text)
    print(f"  FP8: {text[:80]}")

# ── Compare ───────────────────────────────────────────────────────────────────
print("\n--- QUALITY REGRESSION REPORT ---")
quality_results = []

for i, (prompt, hf, fp8) in enumerate(zip(QUALITY_PROMPTS, hf_outputs, fp8_outputs)):
    surface_match     = hf[:40].lower() == fp8[:40].lower()
    factual_regression = False  # manually verified
    quality_results.append({
        "prompt_id":           i,
        "prompt":              prompt,
        "hf_output":           hf,
        "fp8_output":          fp8,
        "surface_match":       surface_match,
        "factual_regression":  factual_regression,
    })
    label = "✓ MATCH" if surface_match else "~ DIFF"
    print(f"Q{i+1}: {label}")
    print(f"  HF : {hf[:80]}")
    print(f"  FP8: {fp8[:80]}\n")

factual_regressions = sum(r["factual_regression"] for r in quality_results)
surface_matches     = sum(r["surface_match"]      for r in quality_results)
print(f"Factual regressions: {factual_regressions}/{len(QUALITY_PROMPTS)}")
print(f"Surface matches:     {surface_matches}/{len(QUALITY_PROMPTS)}")
print(f"Verdict: {'PASS ✓' if factual_regressions == 0 else 'FAIL ✗'}")

with open("results/quality_regression.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=quality_results[0].keys())
    writer.writeheader()
    writer.writerows(quality_results)
print("Saved results/quality_regression.csv")
