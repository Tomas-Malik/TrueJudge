import pandas as pd
import re
from grading import grader
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit this section to switch judges / datasets / experiments
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "/home/tmalik6/Summer/dedup/Summer_Code"

# Name of the column in every CSV that contains the raw judge output.
JUDGE_COL = "Judge Q3"

# Human-readable label for the judge model (used in printed output and summary CSV).
JUDGE_LABEL = "Qwen 3 8B"

# Output file for the summary CSV.
SUMMARY_CSV = "binary_judge_summary_Q3.csv"

# ── Generator experiments ─────────────────────────────────────────────────────
# Each entry defines one generator model to evaluate.
# Keys:
#   key        — short identifier (unused at runtime, just for readability)
#   label      — human-readable generator name printed in output
#   gsm8k_csv  — path to the GSM8K results CSV for this generator
#   math500_csv — path to the MATH-500 results CSV for this generator
#
# Add, remove, or comment out rows to change which generators are evaluated.
# Set either csv path to None to skip that dataset for a given generator.

EXPERIMENTS = [
    {
        "key":         "exp1",
        "label":       "Llama-3.1-8B",
        "gsm8k_csv":   f"/home/tmalik6/Summer/dedup/Summer_Code/GSM8K_q3_binary_eval_exp1_llama8b.csv",
        "math500_csv": f"/home/tmalik6/Summer/dedup/Summer_Code/MATH500_q3_binary_eval_exp1_llama8b.csv",
    },
    {
        "key":         "exp2",
        "label":       "Gemma-2-9B",
        "gsm8k_csv":   f"{RESULTS_DIR}/GSM8K_q3_binary_eval_exp2_gemma2_9b.csv",
        "math500_csv": f"{RESULTS_DIR}/MATH500_q3_binary_eval_exp2_gemma2_9b.csv",
    },
    {
        "key":         "exp3",
        "label":       "Llama-3.3-70B",
        "gsm8k_csv":   f"{RESULTS_DIR}/GSM8K_q3_binary_eval_exp3_llama70b.csv",
        "math500_csv": f"{RESULTS_DIR}/MATH500_q3_binary_eval_exp3_llama70b.csv",
    },
]

# ── Oracle condition ───────────────────────────────────────────────────────────
# Judge is shown the gold solution (always correct by construction).
# Set either csv path to None to skip that dataset for the oracle condition.
# Set INCLUDE_ORACLE = False to skip the oracle entirely.

INCLUDE_ORACLE = True
ORACLE = {
    "key":         "oracle",
    "label":       "True Solution (oracle)",
    "gsm8k_csv":   f"{RESULTS_DIR}/GSM8K_q3_binary_eval_target.csv",
    "math500_csv": f"{RESULTS_DIR}/MATH500_q3_binary_eval_target.csv",
}

# ── Column names in the CSVs ───────────────────────────────────────────────────
# "Solution" column: the generator's raw solution text (used for experiments).
# "Target" column: the gold solution text (used for the oracle condition).

SOLUTION_COL = "Solution"
TARGET_COL   = "Target"

# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def extract_final_answer(model_resp) -> float:
    model_resp = str(model_resp).replace(",", "")
    extracted_num = re.findall(r"-?\d+\.?\d*", model_resp)
    return float(extracted_num[-1]) if extracted_num else "NA"

def extract_gsm8k_ans(true_ans) -> float:
    true_ans = str(true_ans).replace(",", "")
    match = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', true_ans)
    if match:
        s = match.group(1)
        return float(s) if '.' in s else int(s)
    return "NA"

def extract_math_ans(true_ans: str) -> str:
    true_ans = str(true_ans)
    init = len(true_ans) - 1
    targ_len = len("\\boxed")
    final_ans = ""
    while (init - targ_len) >= 0:
        if true_ans[init - targ_len: init] == "\\boxed":
            break
        init -= 1
    if init - targ_len < 0:
        return "NA"
    start = init
    while start < len(true_ans) and true_ans[start] != "{":
        start += 1
    if start == len(true_ans):
        return "NA"
    start += 1
    brackets = 1
    for i in range(start, len(true_ans)):
        if true_ans[i] == "{":
            brackets += 1
        elif true_ans[i] == "}":
            brackets -= 1
            if brackets == 0:
                return final_ans
        final_ans += true_ans[i]
    return "NA"

# ══════════════════════════════════════════════════════════════════════════════
#  COMPARATORS
# ══════════════════════════════════════════════════════════════════════════════

def num_compare(a, b) -> bool:
    try:    return float(a) == float(b)
    except: return False

def math_compare(a, b) -> bool:
    if a in ("NA", "fail", None): return False
    return grader.grade_answer(str(a), str(b))

def always_correct(a, b) -> bool:
    # Oracle: solution shown to the judge IS the gold answer, correct by construction.
    return True

# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE OUTPUT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_binary_judge_output(judge_text: str) -> str:
    match = re.search(r'"verdict"\s*:\s*"([^"]+)"', str(judge_text))
    if match:
        val = match.group(1).strip().lower()
        if val in ("correct", "incorrect"):
            return val
    return "fail"

# ══════════════════════════════════════════════════════════════════════════════
#  CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_binary(dataset_name, generator_label, solutions, judge_raws, gt_answers, extractor, comparator):
    l = len(solutions)

    actual_correct = actual_incorrect = 0
    verdict_correct = verdict_incorrect = parse_fail = 0
    tp = fn = fp = tn = agree = 0

    for soln_raw, judge_raw, gt in zip(solutions, judge_raws, gt_answers):
        pred_ans = extractor(soln_raw)
        is_actually_correct = comparator(pred_ans, gt)

        if is_actually_correct:
            actual_correct += 1
        else:
            actual_incorrect += 1

        verdict = parse_binary_judge_output(judge_raw)

        if verdict == "correct":
            verdict_correct += 1
            if is_actually_correct:
                tp += 1; agree += 1
            else:
                fp += 1
        elif verdict == "incorrect":
            verdict_incorrect += 1
            if is_actually_correct:
                fn += 1
            else:
                tn += 1; agree += 1
        else:
            parse_fail += 1

    accuracy = agree / l if l > 0 else None

    print(f"\n{'═' * 60}")
    print(f"  {dataset_name}  —  Judge: {JUDGE_LABEL}  —  Generator: {generator_label}")
    print(f"{'═' * 60}")
    print(f"  Total questions               : {l}")

    print(f"\n  ── Ground truth breakdown ─────────────────────────────")
    print(f"  Generator actually correct    : {actual_correct}  ({actual_correct/l*100:.1f}%)")
    print(f"  Generator actually incorrect  : {actual_incorrect}  ({actual_incorrect/l*100:.1f}%)")

    print(f"\n  ── Judge verdict distribution ─────────────────────────")
    print(f"  Verdict = correct             : {verdict_correct}  ({verdict_correct/l*100:.1f}%)")
    print(f"  Verdict = incorrect           : {verdict_incorrect}  ({verdict_incorrect/l*100:.1f}%)")
    print(f"  Parse fail                    : {parse_fail}  ({parse_fail/l*100:.1f}%)")

    print(f"\n  ── Confusion matrix (parse-fail counted as wrong) ─────")
    print(f"  TP (actual✓, judge✓)          : {tp}")
    print(f"  FN (actual✓, judge✗ or fail)  : {fn + parse_fail}")
    print(f"  FP (actual✗, judge✓)          : {fp}")
    print(f"  TN (actual✗, judge✗)          : {tn}")

    print(f"\n  ── Judge accuracy (parse-fail = wrong) ────────────────")
    print(f"  Accuracy                      : {agree}/{l}  ({accuracy*100:.1f}%)")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision (verdict=correct)   : {tp}/{tp+fp}  ({precision*100:.1f}%)")
    if tp + fn + parse_fail > 0:
        recall = tp / (tp + fn + parse_fail)
        print(f"  Recall (verdict=correct)      : {tp}/{tp+fn+parse_fail}  ({recall*100:.1f}%)")

    return {
        "dataset":          dataset_name,
        "judge":            JUDGE_LABEL,
        "generator":        generator_label,
        "n":                l,
        "actual_correct":   actual_correct,
        "verdict_correct":  verdict_correct,
        "verdict_incorrect": verdict_incorrect,
        "parse_fail":       parse_fail,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy":         accuracy,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  PER-DATASET WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_gsm8k(csv_path, generator_label, gt_answers):
    df = pd.read_csv(csv_path)
    return analyse_binary(
        dataset_name="GSM8K",
        generator_label=generator_label,
        solutions=df[SOLUTION_COL].to_list(),
        judge_raws=df[JUDGE_COL].to_list(),
        gt_answers=gt_answers,
        extractor=extract_final_answer,
        comparator=num_compare,
    )

def analyse_math500(csv_path, generator_label, gt_answers):
    df = pd.read_csv(csv_path)
    def extractor(raw):
        ans = extract_math_ans(str(raw))
        return "NA" if ans == "NA" else ans
    return analyse_binary(
        dataset_name="MATH500",
        generator_label=generator_label,
        solutions=df[SOLUTION_COL].to_list(),
        judge_raws=df[JUDGE_COL].to_list(),
        gt_answers=gt_answers,
        extractor=extractor,
        comparator=math_compare,
    )

def analyse_gsm8k_oracle(csv_path, generator_label):
    df = pd.read_csv(csv_path)
    solutions = df[TARGET_COL].to_list()
    return analyse_binary(
        dataset_name="GSM8K",
        generator_label=generator_label,
        solutions=solutions,
        judge_raws=df[JUDGE_COL].to_list(),
        gt_answers=[None] * len(solutions),
        extractor=lambda raw: raw,
        comparator=always_correct,
    )

def analyse_math500_oracle(csv_path, generator_label):
    df = pd.read_csv(csv_path)
    solutions = df[TARGET_COL].to_list()
    return analyse_binary(
        dataset_name="MATH500",
        generator_label=generator_label,
        solutions=solutions,
        judge_raws=df[JUDGE_COL].to_list(),
        gt_answers=[None] * len(solutions),
        extractor=lambda raw: raw,
        comparator=always_correct,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

print("Loading GSM8K ground truth...")
gsm8k_ds = load_dataset("openai/gsm8k", "main")
gsm8k_gt = [extract_gsm8k_ans(a) for a in gsm8k_ds['test']['answer']]

print("Loading MATH-500 ground truth...")
math500_ds = load_dataset("HuggingFaceH4/MATH-500")
math500_gt = [extract_math_ans(str(s)) for s in math500_ds['test']['solution']]

all_results = []

for exp in EXPERIMENTS:
    if exp.get("gsm8k_csv"):
        all_results.append(analyse_gsm8k(exp["gsm8k_csv"], exp["label"], gsm8k_gt))
    if exp.get("math500_csv"):
        all_results.append(analyse_math500(exp["math500_csv"], exp["label"], math500_gt))

if INCLUDE_ORACLE:
    if ORACLE.get("gsm8k_csv"):
        all_results.append(analyse_gsm8k_oracle(ORACLE["gsm8k_csv"], ORACLE["label"]))
    if ORACLE.get("math500_csv"):
        all_results.append(analyse_math500_oracle(ORACLE["math500_csv"], ORACLE["label"]))

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'═' * 60}")
print(f"  SUMMARY — Judge: {JUDGE_LABEL}")
print(f"{'═' * 60}")
summary_df = pd.DataFrame(all_results)
summary_df["actual_correct_pct"] = (summary_df["actual_correct"] / summary_df["n"] * 100).round(1)
summary_df["accuracy_pct"]       = (summary_df["accuracy"] * 100).round(1)
print(summary_df[["dataset", "judge", "generator", "n", "actual_correct_pct", "accuracy_pct", "parse_fail"]].to_string(index=False))

summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"\nSummary saved to {SUMMARY_CSV}")