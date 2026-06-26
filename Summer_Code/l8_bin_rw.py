import pandas as pd
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
    max_tokens=2048,
)

def run_judge(prompt: str) -> str:
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful and impartial mathematical evaluator.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
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

# ── Load questions ─────────────────────────────────────────────────────────────

print("Loading GSM8K dataset...")
gsm8k_ds       = load_dataset("openai/gsm8k", "main")
questions_gsm8k = gsm8k_ds['test']['question']
print("Dataset loaded.")

# ── Helper ────────────────────────────────────────────────────────────────────

def run_eval(questions, solutions, prompt_template, label, out_csv):
    judge_outputs = []
    for idx, (q, soln) in enumerate(zip(questions, solutions)):
        print(f"{label} {idx + 1} / {len(questions)}")
        prompt = prompt_template.format(question=q, solution=soln)
        result = run_judge(prompt)
        judge_outputs.append(result)

    pd.DataFrame({
        "Question": list(questions),
        "Solution": list(solutions),
        "Judge L8": judge_outputs,
    }).to_csv(out_csv, index=False)
    print(f"{label} done -> {out_csv}")

# ── Experiment: GSM8K (L70 Rewritten) ────────────────────────────────────────

df = pd.read_csv("/home/tmalik6/Summer/dedup/Summer_Code/GSM8K_l70_rewritten.csv")
solutions = df["L70_Rewritten"].to_list()

run_eval(
    questions_gsm8k, solutions, JUDGE_PROMPT_GSM8K,
    "EXP3 GSM8K (Llama3.3-70B Rewritten)", "GSM8K_l8_binary_eval_exp3_llama70b_rewritten.csv",
)