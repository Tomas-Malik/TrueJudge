import re
import pandas as pd
from datasets import load_dataset
from groq import Groq

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit this section to switch datasets / columns / outputs
# ══════════════════════════════════════════════════════════════════════════════

# ── Groq setup ────────────────────────────────────────────────────────────────

GROQ_API_KEY_PATH = "/home/tmalik6/Summer/dedup/groq_api.txt"
MODEL_NAME        = "llama-3.3-70b-versatile"
TEMPERATURE       = 0.3   # low for faithful rewriting
SEED              = 42
MAX_TOKENS        = 1024

# ── Which dataset to process ──────────────────────────────────────────────────
# Set to "gsm8k" or "math500"

DATASET = "gsm8k"   # <-- change to "math500" to process MATH-500

# ── Input CSVs ────────────────────────────────────────────────────────────────
# CSV containing the L70 solutions to rewrite.

GSM8K_L70_CSV      = "/home/tmalik6/Summer/dedup/Single_Models/GSM8K_llama70B_full_2k_reordered.csv"
GSM8K_L70_COL      = "Baseline Full"   # column in the above CSV with L70's raw solution

MATH500_L70_CSV    = "/home/tmalik6/Summer/dedup/Single_Models/MATH500_full_L70.csv"
MATH500_L70_COL    = "Llama70B"        # column in the above CSV with L70's raw solution

# ── Output CSVs ───────────────────────────────────────────────────────────────
# Output will have columns: Question, L70_Original, True_Solution, L70_Rewritten

GSM8K_OUT_CSV      = "GSM8K_l70_rewritten.csv"
MATH500_OUT_CSV    = "MATH500_l70_rewritten.csv"

# ── Optional: limit rows for testing (set to None to process all) ─────────────

ROW_LIMIT = None   # e.g. 10 for a quick test run

# ══════════════════════════════════════════════════════════════════════════════
#  REWRITE PROMPTS
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
To find out how much Janet makes every day at the farmers' market, let's break down the problem step by step.

Step 1: Calculate the total number of eggs laid by Janet's ducks per day.
Janet's ducks lay 16 eggs per day.

Step 2: Determine the number of eggs Janet eats for breakfast.
She eats 3 eggs for breakfast every morning.

Step 3: Calculate the number of eggs Janet uses for baking muffins.
She uses 4 eggs to bake muffins for her friends every day.

Step 4: Calculate the total number of eggs used by Janet (for breakfast and baking).
Total eggs used = eggs for breakfast + eggs for baking = 3 + 4 = 7 eggs.

Step 5: Calculate the number of eggs Janet has left to sell at the farmers' market.
Eggs left to sell = total eggs laid - total eggs used = 16 - 7 = 9 eggs.

Step 6: Determine the price per egg that Janet sells at the farmers' market.
She sells each egg for $2.

Step 7: Calculate the total amount of money Janet makes every day at the farmers' market.
Total money made = number of eggs sold * price per egg = 9 * $2 = $18.

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
{l70_solution}

Rewritten solution:"""

REWRITE_PROMPT_MATH500 = """\
You are a style editor. You will be given a mathematical reasoning solution and a reference solution to the same problem. Answers may use LaTeX notation.

Your task is to rewrite the given solution so that it matches the LENGTH and STYLE of the reference solution as closely as possible — same level of brevity, same format, same notation conventions.

CRITICAL rules:
- Do NOT fix any mathematical errors. If the given solution is wrong, keep it wrong.
- Do NOT add reasoning steps that are not in the given solution.
- Do NOT remove reasoning steps that are present in the given solution.
- Collapse verbose "Step N:" structure into compact prose with inline LaTeX, exactly as the reference does.
- The final answer must appear inside \\boxed{} if the reference uses that convention.
- Output ONLY the rewritten solution, nothing else. No preamble, no explanation.

Reference solution (style target):
{true_solution}

Solution to rewrite:
{l70_solution}

Rewritten solution:"""

# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT
# ══════════════════════════════════════════════════════════════════════════════

with open(GROQ_API_KEY_PATH, "r") as f:
    api_key = f.read().strip()

client = Groq(api_key=api_key)

def call_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        seed=SEED,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT CLEANER
# ══════════════════════════════════════════════════════════════════════════════

# Strips common preamble/postamble the model may add despite being told not to.

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
#  CORE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_rewrite(questions, l70_solutions, true_solutions, prompt_template, label, out_csv):
    assert len(questions) == len(l70_solutions) == len(true_solutions), \
        "Mismatch in list lengths — check your CSVs and dataset alignment."

    n = len(questions) if ROW_LIMIT is None else min(ROW_LIMIT, len(questions))

    rewritten = []
    for idx in range(n):
        print(f"{label}  {idx + 1} / {n}")
        prompt = prompt_template.format(
            true_solution=true_solutions[idx],
            l70_solution=l70_solutions[idx],
        )
        result = clean_rewritten(call_groq(prompt))
        rewritten.append(result)

    pd.DataFrame({
        "Question":      list(questions[:n]),
        "L70_Original":  list(l70_solutions[:n]),
        "True_Solution": list(true_solutions[:n]),
        "L70_Rewritten": rewritten,
    }).to_csv(out_csv, index=False, quoting=1)   # quoting=1 = QUOTE_ALL (safe for multi-line cells)

    print(f"\n{label} done -> {out_csv}")

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET-SPECIFIC ENTRYPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def process_gsm8k():
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main")
    questions      = ds['test']['question']
    true_solutions = ds['test']['answer']

    df_l70 = pd.read_csv(GSM8K_L70_CSV)
    l70_solutions = df_l70[GSM8K_L70_COL].to_list()

    run_rewrite(
        questions, l70_solutions, true_solutions,
        REWRITE_PROMPT_GSM8K,
        label="GSM8K rewrite",
        out_csv=GSM8K_OUT_CSV,
    )

def process_math500():
    print("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500")
    questions      = ds['test']['problem']
    true_solutions = ds['test']['solution']

    df_l70 = pd.read_csv(MATH500_L70_CSV)
    l70_solutions = df_l70[MATH500_L70_COL].to_list()

    run_rewrite(
        questions, l70_solutions, true_solutions,
        REWRITE_PROMPT_MATH500,
        label="MATH500 rewrite",
        out_csv=MATH500_OUT_CSV,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

if DATASET == "gsm8k":
    process_gsm8k()
elif DATASET == "math500":
    process_math500()
else:
    raise ValueError(f"Unknown DATASET value: '{DATASET}'. Must be 'gsm8k' or 'math500'.")