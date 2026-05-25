"""
Script 5: PyTorch Profiler
===========================
Profiles two inference cases to identify compute bottlenecks:
  Case 1 — Prefill-heavy: 1500 token prompt, 32 output tokens
  Case 2 — Decode-heavy:  short prompt, 256 output tokens

Saves Chrome traces to results/trace_prefill_heavy.json
                     and results/trace_decode_heavy.json

Open traces at chrome://tracing for visual timeline.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
os.makedirs("results", exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda").eval()
print("Model loaded.")

# ── Warmup ────────────────────────────────────────────────────────────────────
warmup_inputs = tokenizer("Hello.", return_tensors="pt").to("cuda")
for _ in range(3):
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()
print("Warmup done.")

# ── Case 1: Prefill-heavy ─────────────────────────────────────────────────────
long_prompt  = "Explain the history of artificial intelligence. " * 40
inputs_long  = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=1500).to("cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof_prefill:
    with record_function("prefill_heavy"):
        with torch.no_grad():
            _ = model.generate(**inputs_long, max_new_tokens=32, do_sample=False)

torch.cuda.synchronize()
print("\n--- CASE 1: PREFILL-HEAVY ---")
print(prof_prefill.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof_prefill.export_chrome_trace("results/trace_prefill_heavy.json")

# ── Case 2: Decode-heavy ──────────────────────────────────────────────────────
inputs_short = tokenizer("Tell me a story.", return_tensors="pt").to("cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof_decode:
    with record_function("decode_heavy"):
        with torch.no_grad():
            _ = model.generate(**inputs_short, max_new_tokens=256, do_sample=False)

torch.cuda.synchronize()
print("\n--- CASE 2: DECODE-HEAVY ---")
print(prof_decode.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof_decode.export_chrome_trace("results/trace_decode_heavy.json")

print("\nTraces saved. Open in chrome://tracing")

# ── Key finding ───────────────────────────────────────────────────────────────
# ── Auto-compute and print key finding from actual profiler output ────────────
cpu_prefill_ms  = sum(e.cpu_time_total for e in prof_prefill.key_averages()) / 1000
cuda_prefill_ms = sum(e.cuda_time_total for e in prof_prefill.key_averages()) / 1000
cpu_decode_ms   = sum(e.cpu_time_total  for e in prof_decode.key_averages())  / 1000
cuda_decode_ms  = sum(e.cuda_time_total for e in prof_decode.key_averages())  / 1000

ratio_prefill = cpu_prefill_ms  / cuda_prefill_ms  if cuda_prefill_ms  else 0
ratio_decode  = cpu_decode_ms   / cuda_decode_ms   if cuda_decode_ms   else 0
idle_pct      = (1 - cuda_decode_ms / cpu_decode_ms) * 100 if cpu_decode_ms else 0

print(f"""
KEY FINDING (measured):
  Prefill-heavy CPU/CUDA ratio: {ratio_prefill:.2f}x
  Decode-heavy  CPU/CUDA ratio: {ratio_decode:.2f}x
  H100 idle ~{idle_pct:.0f}% of decode time (Python kernel dispatch overhead)
  This CPU-bound bottleneck is what vLLM's C++ scheduler eliminates.
""")
