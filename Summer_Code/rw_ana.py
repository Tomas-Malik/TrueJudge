import pandas as pd
import re
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "/home/tmalik6/Summer/dedup/Summer_Code"

SUMMARY_CSV = "binary_judge_summary_rewritten_gsm8k.csv"

# Each judge: label, the CSV containing its output, and the column with raw output.
JUDGES = [
    {
        "label":      "Llama-3.1-8B",
        "csv":        f"{RESULTS_DIR}/GSM8K_l8_binary_eval_exp3_llama70b_rewritten.csv",
        "judge_col":  "Judge L8",
    },
    {
        "label":      "Gemma-2-9B",
        "csv":        f"{RESULTS_DIR}/GSM8K_g9_binary_eval_exp3_llama70b_rewritten.csv",
        "judge_col":  "Judge G9",
    },
    {
        "label":      "Qwen3-8B",
        "csv":        f"{RESULTS_DIR}/GSM8K_q3_binary_eval_exp3_llama70b_rewritten.csv",
        "judge_col":  "Judge Q3",
    },
]

# Column in each CSV that holds the solution text shown to the judge.
SOLUTION_COL = "Solution"

# Generator being judged (for display only).
GENERATOR_LABEL = "Llama-3.3-70B (Rewritten)"

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

# ══════════════════════════════════════════════════════════════════════════════
#  COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════

def num_compare(a, b) -> bool:
    try:    return float(a) == float(b)
    except: return False

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

def analyse_binary(judge_label, solutions, judge_raws, gt_answers):
    l = len(solutions)

    actual_correct = actual_incorrect = 0
    verdict_correct = verdict_incorrect = parse_fail = 0
    tp = fn = fp = tn = agree = 0

    for soln_raw, judge_raw, gt in zip(solutions, judge_raws, gt_answers):
        pred_ans = extract_final_answer(soln_raw)
        is_actually_correct = num_compare(pred_ans, gt)

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
    print(f"  GSM8K  —  Judge: {judge_label}  —  Generator: {GENERATOR_LABEL}")
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
        "dataset":            "GSM8K",
        "judge":              judge_label,
        "generator":          GENERATOR_LABEL,
        "n":                  l,
        "actual_correct":     actual_correct,
        "verdict_correct":    verdict_correct,
        "verdict_incorrect":  verdict_incorrect,
        "parse_fail":         parse_fail,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy":           accuracy,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

print("Loading GSM8K ground truth...")
gsm8k_ds = load_dataset("openai/gsm8k", "main")
gsm8k_gt = [extract_gsm8k_ans(a) for a in gsm8k_ds['test']['answer']]
print("Dataset loaded.")

all_results = []

for judge in JUDGES:
    df = pd.read_csv(judge["csv"])
    result = analyse_binary(
        judge_label=judge["label"],
        solutions=df[SOLUTION_COL].to_list(),
        judge_raws=df[judge["judge_col"]].to_list(),
        gt_answers=gsm8k_gt,
    )
    all_results.append(result)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'═' * 60}")
print(f"  SUMMARY — Generator: {GENERATOR_LABEL}")
print(f"{'═' * 60}")
summary_df = pd.DataFrame(all_results)
summary_df["actual_correct_pct"] = (summary_df["actual_correct"] / summary_df["n"] * 100).round(1)
summary_df["accuracy_pct"]       = (summary_df["accuracy"] * 100).round(1)
print(summary_df[["dataset", "judge", "generator", "n", "actual_correct_pct", "accuracy_pct", "parse_fail"]].to_string(index=False))

summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"\nSummary saved to {SUMMARY_CSV}")