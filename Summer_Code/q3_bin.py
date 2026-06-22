import pandas as pd
from datasets import load_dataset
from vllm import LLM
from vllm.sampling_params import SamplingParams

# ── vLLM engine ───────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B"

llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.5,
    seed=42,
    max_tokens=2048,
)

def run_judge(prompt: str) -> str:
    formatted = (
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

# ── Prompt ────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_GSM8K = """\
    Question:
    {question}

    Proposed solution:
    {solution}

    Evaluate whether the proposed solution is correct.
    Conclude with a JSON in Markdown format:
```json
{{
    "verdict": "correct" or "incorrect"
}}
```
"""

JUDGE_PROMPT_MATH500 = """\
    Question:
    {question}

    This is a mathematical reasoning problem. Answers may be expressed in LaTeX notation.

    Proposed solution:
    {solution}

    Evaluate whether the proposed solution is correct.
    Conclude with a JSON in Markdown format:
```json
{{
    "verdict": "correct" or "incorrect"
}}
```
"""

# ── Load CSVs (questions / shared base) ────────────────────────────────────────

df_gsm8k = pd.read_csv("/home/tmalik6/Summer/dedup/Single_Models/GSM8K_Gemma2_9B_SC3_temp05_full_2k.csv")
questions_gsm8k = df_gsm8k["Question"].to_list()

df_math500 = pd.read_csv("/home/tmalik6/Summer/dedup/Single_Models/MATH500_gemmav2.csv")

# ── Load datasets ─────────────────────────────────────────────────────────────

print("Loading GSM8K dataset...")
gsm8k_ds         = load_dataset("openai/gsm8k", "main")
problems_gsm8k   = gsm8k_ds['test']['question']
full_solns_gsm8k = gsm8k_ds['test']['answer']

print("Loading MATH-500 dataset...")
math500_ds         = load_dataset("HuggingFaceH4/MATH-500")
problems_math500   = math500_ds['test']['problem']
full_solns_math500 = math500_ds['test']['solution']

print("All datasets loaded.")

# ── EXPERIMENT CONFIG ───────────────────────────────────────────────────────────
# Fill in the CSV paths and the column name containing the generated
# (model) response/solution text for each experiment.

# Experiment 1: Judge Llama-3.1-8B's response
EXP1_GSM8K_CSV    = "/home/tmalik6/Summer/dedup/Single_Models/GSM8K_Llama8B_SC3_temp05_full_2k.csv"
EXP1_GSM8K_COL    = "Llama8B SC (1)"
EXP1_MATH500_CSV  = "/home/tmalik6/Summer/dedup/Single_Models/MATH500_l8_sc_full.csv"
EXP1_MATH500_COL  = "L8 1"

# Experiment 2: Judge Gemma-2-9B's own response
EXP2_GSM8K_CSV    = "/home/tmalik6/Summer/dedup/Single_Models/GSM8K_Gemma2_9B_SC3_temp05_full_2k.csv"
EXP2_GSM8K_COL    = "Gemma2_9B SC (1)"
EXP2_MATH500_CSV  = "/home/tmalik6/Summer/dedup/Single_Models/MATH500_gemmav2.csv"
EXP2_MATH500_COL  = "Gemma"

# Experiment 3: Judge Llama-3.3-70B's response
EXP3_GSM8K_CSV    = "/home/tmalik6/Summer/dedup/Single_Models/GSM8K_llama70B_full_2k_reordered.csv"
EXP3_GSM8K_COL    = "Baseline Full"
EXP3_MATH500_CSV  = "/home/tmalik6/Summer/dedup/Single_Models/MATH500_full_L70.csv"
EXP3_MATH500_COL  = "Llama70B"

# ── Helper to run an evaluation pass ────────────────────────────────────────────

def run_eval(questions, solutions, prompt_template, label, out_csv, solution_col="Solution"):
    judge_outputs = []
    for idx, (q, soln) in enumerate(zip(questions, solutions)):
        print(f"{label} {idx + 1} / {len(questions)}")
        prompt = prompt_template.format(question=q, solution=soln)
        result = run_judge(prompt)
        judge_outputs.append(result)

    pd.DataFrame({
        "Question": list(questions),
        solution_col: list(solutions),
        "Judge Q3": judge_outputs,
    }).to_csv(out_csv, index=False)
    print(f"{label} done -> {out_csv}")

# ── Experiment 1: GSM8K ──────────────────────────────────────────────────────────

df_exp1_gsm8k = pd.read_csv(EXP1_GSM8K_CSV)
solutions_exp1_gsm8k = df_exp1_gsm8k[EXP1_GSM8K_COL].to_list()
run_eval(
    questions_gsm8k, solutions_exp1_gsm8k, JUDGE_PROMPT_GSM8K,
    "EXP1 GSM8K (Llama8B)", "GSM8K_q3_binary_eval_exp1_llama8b.csv",
)

# ── Experiment 1: MATH500 ────────────────────────────────────────────────────────

df_exp1_math500 = pd.read_csv(EXP1_MATH500_CSV)
solutions_exp1_math500 = df_exp1_math500[EXP1_MATH500_COL].to_list()
run_eval(
    problems_math500, solutions_exp1_math500, JUDGE_PROMPT_MATH500,
    "EXP1 MATH500 (Llama8B)", "MATH500_q3_binary_eval_exp1_llama8b.csv",
)

# ── Experiment 2: GSM8K ──────────────────────────────────────────────────────────

df_exp2_gsm8k = pd.read_csv(EXP2_GSM8K_CSV)
solutions_exp2_gsm8k = df_exp2_gsm8k[EXP2_GSM8K_COL].to_list()
run_eval(
    questions_gsm8k, solutions_exp2_gsm8k, JUDGE_PROMPT_GSM8K,
    "EXP2 GSM8K (Gemma2-9B self)", "GSM8K_q3_binary_eval_exp2_gemma2_9b.csv",
)

# ── Experiment 2: MATH500 ────────────────────────────────────────────────────────

df_exp2_math500 = pd.read_csv(EXP2_MATH500_CSV)
solutions_exp2_math500 = df_exp2_math500[EXP2_MATH500_COL].to_list()
run_eval(
    problems_math500, solutions_exp2_math500, JUDGE_PROMPT_MATH500,
    "EXP2 MATH500 (Gemma2-9B self)", "MATH500_q3_binary_eval_exp2_gemma2_9b.csv",
)

# ── Experiment 3: GSM8K ──────────────────────────────────────────────────────────

df_exp3_gsm8k = pd.read_csv(EXP3_GSM8K_CSV)
solutions_exp3_gsm8k = df_exp3_gsm8k[EXP3_GSM8K_COL].to_list()
run_eval(
    questions_gsm8k, solutions_exp3_gsm8k, JUDGE_PROMPT_GSM8K,
    "EXP3 GSM8K (Llama3.3-70B)", "GSM8K_q3_binary_eval_exp3_llama70b.csv",
)

# ── Experiment 3: MATH500 ────────────────────────────────────────────────────────

df_exp3_math500 = pd.read_csv(EXP3_MATH500_CSV)
solutions_exp3_math500 = df_exp3_math500[EXP3_MATH500_COL].to_list()
run_eval(
    problems_math500, solutions_exp3_math500, JUDGE_PROMPT_MATH500,
    "EXP3 MATH500 (Llama3.3-70B)", "MATH500_q3_binary_eval_exp3_llama70b.csv",
)

# ── Experiment 4 (Target): GSM8K ─────────────────────────────────────────────────
# True/gold solution judgment — not yet run with Gemma2-9B (only exists for L8 so far)

run_eval(
    questions_gsm8k, full_solns_gsm8k, JUDGE_PROMPT_GSM8K,
    "TARGET GSM8K (gold solution)", "GSM8K_q3_binary_eval_target.csv",
    solution_col="Target",
)

# ── Experiment 4 (Target): MATH500 ───────────────────────────────────────────────

run_eval(
    problems_math500, full_solns_math500, JUDGE_PROMPT_MATH500,
    "TARGET MATH500 (gold solution)", "MATH500_q3_binary_eval_target.csv",
    solution_col="Target",
)