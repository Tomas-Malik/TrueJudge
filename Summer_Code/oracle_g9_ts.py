import pandas as pd
import re
import time
from typing import Optional, Dict
from datasets import load_dataset
from vllm import LLM
from vllm.sampling_params import SamplingParams

# ── vLLM engine ───────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.5,
    seed=42,
    max_tokens=6000,
)

def run_judge(prompt: str) -> str:
    """Run a single prompt through the local Llama-3.1-8B-Instruct judge."""
    # Llama-3 instruct chat template
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful and impartial judge.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

# ── Shared prompt suffix ──────────────────────────────────────────────────────

JUDGE_OPTIONS = """
    Compare both solutions carefully and decide which is correct.
    Conclude with a JSON in Markdown format indicating your choice
    between "solution_1", "solution_2", "none_correct" or "both_correct":
```json
    {{
    "answer": "..."
    }}
```
    """

# ── Load model responses from CSVs ────────────────────────────────────────────

# GSM8K — only Gemma needed from CSV; target comes from HuggingFace
df_gsm8k = pd.read_csv("/home/tmalik6/LLMR/Code/GSM8K/CSVs_latest/GSM8K_Gemma2_9B_SC3_temp05_full_2k.csv")
g9_1_gsm8k      = df_gsm8k["Gemma2_9B SC (1)"].to_list()
questions_gsm8k = df_gsm8k["Question"].to_list()

# MATH500 — only Gemma needed from CSV; target comes from HuggingFace
df_math500 = pd.read_csv("/home/tmalik6/LLMR/Code/math500/CSVs_latest/MATH500_gemmav2.csv")
g9_1_math500      = df_math500["Gemma"].to_list()
# questions come from HuggingFace (problems_math500 below)

# ── Load full solutions from HuggingFace ──────────────────────────────────────

print("Loading GSM8K dataset...")
gsm8k_ds         = load_dataset("openai/gsm8k", "main")
full_solns_gsm8k = gsm8k_ds['test']['answer']

print("Loading MATH-500 dataset...")
math500_ds         = load_dataset("HuggingFaceH4/MATH-500")
full_solns_math500 = math500_ds['test']['solution']
problems_math500   = math500_ds['test']['problem']

print("All datasets loaded.")

# ── GSM8K ─────────────────────────────────────────────────────────────────────

judge_gsm8k = []
for idx, (g9, q, full_soln) in enumerate(
    zip(g9_1_gsm8k, questions_gsm8k, full_solns_gsm8k)
):
    print(f"GSM8K {idx + 1} / {len(questions_gsm8k)}")
    user = f"""
    Question:
    {q}

    Solution 1:
    {g9}
    \n _____________
    Solution 2:
    {full_soln}
    \n _____________
    {JUDGE_OPTIONS}"""
    result = run_judge(user)
    judge_gsm8k.append(result)

pd.DataFrame({
    "Question":   questions_gsm8k,
    "G9":         g9_1_gsm8k,
    "Target":     list(full_solns_gsm8k),
    "Judge L8":   judge_gsm8k,
}).to_csv("GSM8K_l8_judge_2way_blind_fullsoln.csv", index=False)
print("GSM8K done")

# ── MATH500 ───────────────────────────────────────────────────────────────────

judge_math500 = []
for idx, (g9, q, full_soln) in enumerate(
    zip(g9_1_math500, problems_math500, full_solns_math500)
):
    print(f"MATH500 {idx + 1} / {len(problems_math500)}")
    user = f"""
    Question:
    {q}

    This is a mathematical reasoning problem. Answers may be expressed in LaTeX notation.

    Solution 1:
    {g9}
    \n _____________
    Solution 2:
    {full_soln}
    \n _____________
    {JUDGE_OPTIONS}"""
    result = run_judge(user)
    judge_math500.append(result)

pd.DataFrame({
    "Question":   list(problems_math500),
    "G9":         g9_1_math500,
    "Target":     list(full_solns_math500),
    "Judge L8":   judge_math500,
}).to_csv("MATH500_l8_judge_2way_blind_fullsoln.csv", index=False)
print("MATH500 done")