import pandas as pd
from datasets import load_dataset
from vllm import LLM
from vllm.sampling_params import SamplingParams

import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# ── vLLM engine ───────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"

llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.5,
    seed=42,
    max_tokens=8192,
)

def run_judge(prompt: str) -> str:
    formatted = (
        "<|im_start|>system\n"
        "You are a helpful and impartial mathematical evaluator.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

# ── Prompts ───────────────────────────────────────────────────────────────────

# Condition 1: binary verdict on a single (gold) solution
PROMPT_BINARY = """\
    Question:
    {question}
    {math_note}
    Proposed solution:
    {solution_a}

    Evaluate whether the proposed solution is correct.
    Conclude with a JSON in Markdown format:
```json
{{
    "verdict": "correct" or "incorrect"
}}
```
"""

# Condition 2: 2-way selection (gold vs gemma)
PROMPT_2WAY = """\
    Question:
    {question}
    {math_note}
    Solution 1:
    {solution_a}

    _____________

    Solution 2:
    {solution_b}

    _____________

    Compare both solutions carefully and decide which is correct.
    Conclude with a JSON in Markdown format indicating your choice
    between "solution_1", "solution_2", "none_correct" or "both_correct":
```json
{{
    "answer": "..."
}}
```
"""

# Condition 3: 3-way selection (gold, llama, gemma)
PROMPT_3WAY = """\
    Question:
    {question}
    {math_note}
    Solution 1:
    {solution_a}

    _____________

    Solution 2:
    {solution_b}

    _____________

    Solution 3:
    {solution_c}

    _____________

    Compare all three solutions carefully and decide which is correct.
    Conclude with a JSON in Markdown format indicating your choice
    between "solution_1", "solution_2", "solution_3", "none_correct" or "multiple_correct":
```json
{{
    "answer": "..."
}}
```
"""

MATH_NOTE = "This is a mathematical reasoning problem. Answers may be expressed in LaTeX notation.\n"

# ── Load CSVs ─────────────────────────────────────────────────────────────────

print("Loading CSVs...")

# GSM8K
df_gsm8k_gemma = pd.read_csv("/home/tmalik6/Summer/dedup/Single_Models/GSM8K_Gemma2_9B_SC3_temp05_full_2k.csv")
df_gsm8k_llama = pd.read_csv("/home/tmalik6/Summer/dedup/Single_Models/GSM8K_Llama8B_SC3_temp05_full_2k.csv")
questions_gsm8k   = df_gsm8k_gemma["Question"].to_list()
gemma_gsm8k       = df_gsm8k_gemma["Gemma2_9B SC (1)"].to_list()
llama_gsm8k       = df_gsm8k_llama["Llama8B SC (1)"].to_list()

# MATH500
df_math500_gemma = pd.read_csv("/home/tmalik6/Summer/dedup/Single_Models/MATH500_gemmav2.csv")
df_math500_llama = pd.read_csv("/home/tmalik6/Summer/dedup/Single_Models/MATH500_l8_sc_full.csv")
gemma_math500    = df_math500_gemma["Gemma"].to_list()
llama_math500    = df_math500_llama["L8 1"].to_list()

# ── Load HuggingFace datasets ─────────────────────────────────────────────────

print("Loading GSM8K dataset...")
gsm8k_ds         = load_dataset("openai/gsm8k", "main")
full_solns_gsm8k = gsm8k_ds['test']['answer']

print("Loading MATH-500 dataset...")
math500_ds         = load_dataset("HuggingFaceH4/MATH-500")
full_solns_math500 = math500_ds['test']['solution']
problems_math500   = math500_ds['test']['problem']

print("All data loaded.")

# ── Helper to run one dataset through all 3 conditions ───────────────────────

def run_all_conditions(tag, questions, gold_solns, gemma_solns, llama_solns, math_note=""):
    c1, c2, c3 = [], [], []
    n = len(questions)

    for idx, (q, gold, gemma, llama) in enumerate(
        zip(questions, gold_solns, gemma_solns, llama_solns)
    ):
        print(f"{tag} {idx + 1} / {n}")

        # Condition 1: gold only, binary verdict
        p1 = PROMPT_BINARY.format(question=q, math_note=math_note, solution_a=gold)
        c1.append(run_judge(p1))

        # Condition 2: gemma (sol 1) vs gold (sol 2), 2-way
        p2 = PROMPT_2WAY.format(question=q, math_note=math_note, solution_a=gemma, solution_b=gold)
        c2.append(run_judge(p2))

        # Condition 3: llama (sol 1), gemma (sol 2), gold (sol 3), 3-way
        p3 = PROMPT_3WAY.format(question=q, math_note=math_note, solution_a=llama, solution_b=gemma, solution_c=gold)
        c3.append(run_judge(p3))

    return c1, c2, c3

# ── GSM8K ─────────────────────────────────────────────────────────────────────

c1_gsm, c2_gsm, c3_gsm = run_all_conditions(
    tag="GSM8K",
    questions=questions_gsm8k,
    gold_solns=list(full_solns_gsm8k),
    gemma_solns=gemma_gsm8k,
    llama_solns=llama_gsm8k,
    math_note="",
)

pd.DataFrame({
    "Question":       questions_gsm8k,
    "Gold":           list(full_solns_gsm8k),
    "Gemma":          gemma_gsm8k,
    "Llama":          llama_gsm8k,
    "C1_Binary":      c1_gsm,
    "C2_2way":        c2_gsm,
    "C3_3way":        c3_gsm,
}).to_csv("GSM8K_qwen3_judge_all_conditions.csv", index=False)
print("GSM8K done")

# ── MATH500 ───────────────────────────────────────────────────────────────────

c1_math, c2_math, c3_math = run_all_conditions(
    tag="MATH500",
    questions=list(problems_math500),
    gold_solns=list(full_solns_math500),
    gemma_solns=gemma_math500,
    llama_solns=llama_math500,
    math_note=MATH_NOTE,
)

pd.DataFrame({
    "Question":       list(problems_math500),
    "Gold":           list(full_solns_math500),
    "Gemma":          gemma_math500,
    "Llama":          llama_math500,
    "C1_Binary":      c1_math,
    "C2_2way":        c2_math,
    "C3_3way":        c3_math,
}).to_csv("MATH500_qwen3_judge_all_conditions.csv", index=False)
print("MATH500 done")