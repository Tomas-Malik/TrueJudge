import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM
from vllm.sampling_params import SamplingParams

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── vLLM setup ────────────────────────────────────────────────────────────────

MODEL_NAME           = "Qwen/Qwen3-8B"
GPU_MEMORY_UTIL      = 0.9
MAX_MODEL_LEN        = 8192
TENSOR_PARALLEL_SIZE = 1
TEMPERATURE          = 0.3
SEED                 = 42
MAX_TOKENS           = 1024

# ── Input CSV ─────────────────────────────────────────────────────────────────

GSM8K_Q3_CSV = "/home/tmalik6/Summer/dedup/Single_Models/GSM8K_qwen8b_full.csv"
GSM8K_Q3_COL = "Qwen"

# ── Output CSV ────────────────────────────────────────────────────────────────
# Columns: Question, Q3_Original, True_Solution, Q3_Rewritten

GSM8K_OUT_CSV = "GSM8K_q3_rewritten.csv"

# ── Optional row limit for test runs (None = process all) ─────────────────────

ROW_LIMIT = None

# ══════════════════════════════════════════════════════════════════════════════
#  REWRITE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

REWRITE_PROMPT_GSM8K = """\
You are a style editor. You will be given a math solution and a reference solution to the same problem.

Your task is to rewrite the given solution so that it matches the LENGTH and STYLE of the reference solution as closely as possible — same level of brevity, same format, same tone.

CRITICAL rules:
- Do NOT fix any mathematical errors. If the given solution is wrong, keep it wrong.
- Do NOT add reasoning steps that are not in the given solution.
- Do NOT remove reasoning steps that are present in the given solution.
- Collapse verbose "Step N:" structure into compact inline arithmetic, exactly as the reference does.
- Output ONLY the rewritten solution, nothing else. No preamble, no explanation.

--- EXAMPLE ---

Reference solution (style target):
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Solution to rewrite:
<think>
Okay, let's see. So the problem is about Janet's ducks laying eggs, and figuring out how much she makes at the farmers' market each day. Let me break it down step by step.

First, the ducks lay 16 eggs every day. That's the starting point. Then, Janet uses some of these eggs for her own consumption and for baking muffins. She eats three eggs for breakfast each morning. So that's three eggs used up. Then she bakes muffins for her friends every day with four eggs. Wait, does that mean she uses four eggs for muffins each day? So total eggs used by her would be three plus four, which is seven eggs per day.

So, if she starts with 16 eggs, subtract the seven she uses, that leaves 16 minus 7, which is 9 eggs. Then, she sells the remainder at the farmers' market for $2 per egg. So 9 eggs times $2 per egg would be 9 times 2, which is $18.

Wait, let me check again. The problem says she eats three for breakfast every morning and bakes muffins with four. So that's 3 + 4 = 7 eggs used daily. Total eggs laid: 16. So 16 - 7 = 9 eggs left. Each sold at $2, so 9 * 2 = $18. Yeah, that seems right.

I think that's all. Let me make sure I didn't miss anything. The question is asking for the daily earnings at the market, so the calculation is correct. No other factors mentioned, like any other expenses or different prices. So the answer should be $18 per day.
</think>

Step 1: Calculate the total number of eggs laid by the ducks daily.
Janet's ducks lay **16 eggs per day**.

Step 2: Determine the number of eggs Janet uses for her own consumption and baking.
She eats **3 eggs for breakfast** and uses **4 eggs for muffins**, totaling **3 + 4 = 7 eggs** used daily.

Step 3: Subtract the eggs used from the total eggs laid to find the remainder for sale.
**16 total eggs - 7 eggs used = 9 eggs remaining**.

Step 4: Calculate the daily earnings from selling the remaining eggs at $2 per egg.
**9 eggs × $2 per egg = $18**.

Final numerical answer: $18

Rewritten solution:
Janet uses 3 + 4 = <<3+4=7>>7 eggs, leaving 16 - 7 = <<16-7=9>>9 eggs to sell.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.
#### 18

--- END EXAMPLE ---

Now apply the same transformation to the following.

Reference solution (style target):
{true_solution}

Solution to rewrite:
{q3_solution}

Rewritten solution:"""

# ══════════════════════════════════════════════════════════════════════════════
#  vLLM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=GPU_MEMORY_UTIL,
    max_model_len=MAX_MODEL_LEN,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    seed=SEED,
    max_tokens=MAX_TOKENS,
)

def run_rewrite_vllm(prompt: str) -> str:
    # Qwen3 ChatML format with /no_think to suppress thinking mode
    formatted = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt} /no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT CLEANER
# ══════════════════════════════════════════════════════════════════════════════

_PREAMBLE = re.compile(
    r"(?i)^("
    r"here'?s?\s+(the\s+)?rewritten\s+solution\s*[:\-]?\s*"
    r"|rewritten\s+solution\s*[:\-]?\s*"
    r"|sure[,!]?\s+here.*?:\s*"
    r"|certainly[,!]?\s+here.*?:\s*"
    r"|below\s+is.*?:\s*"
    r")"
)

_POSTAMBLE = re.compile(
    r"(?i)\s*(i hope this .*|let me know if.*|note[:\s]+(?!.*\d).{0,120})$"
)

def clean_rewritten(text: str) -> str:
    text = _PREAMBLE.sub("", text.strip())
    text = _POSTAMBLE.sub("", text.strip())
    return text.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

print("Loading GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main")
questions      = ds['test']['question']
true_solutions = ds['test']['answer']
print("Dataset loaded.")

df_q3 = pd.read_csv(GSM8K_Q3_CSV)
q3_solutions = df_q3[GSM8K_Q3_COL].to_list()

assert len(questions) == len(q3_solutions), \
    f"Length mismatch: dataset has {len(questions)} questions but CSV has {len(q3_solutions)} rows."

n = len(questions) if ROW_LIMIT is None else min(ROW_LIMIT, len(questions))

rewritten = []
for idx in range(n):
    print(f"GSM8K Q3 rewrite  {idx + 1} / {n}")
    prompt = REWRITE_PROMPT_GSM8K.format(
        true_solution=true_solutions[idx],
        q3_solution=q3_solutions[idx],
    )
    result = clean_rewritten(run_rewrite_vllm(prompt))
    rewritten.append(result)

pd.DataFrame({
    "Question":      list(questions[:n]),
    "Q3_Original":   list(q3_solutions[:n]),
    "True_Solution": list(true_solutions[:n]),
    "Q3_Rewritten":  rewritten,
}).to_csv(GSM8K_OUT_CSV, index=False, quoting=1)

print(f"\nDone -> {GSM8K_OUT_CSV}")