import pandas as pd
import re
from grading import grader
from datasets import load_dataset

# ── Extractors ────────────────────────────────────────────────────────────────

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

def eq(a, b) -> bool:
    return grader.grade_answer(a, b)

# ── Comparators ───────────────────────────────────────────────────────────────

def num_compare(a, b) -> bool:
    try:    return float(a) == float(b)
    except: return False

def math_compare(a, b) -> bool:
    if a in ("NA", "fail", None): return False
    return eq(str(a), str(b))

def always_correct(a, b) -> bool:
    # Oracle condition: the "solution" shown to the judge IS the ground-truth
    # target, so it is correct by construction, regardless of extractor output.
    return True

# ── Judge output parser (binary verdict) ───────────────────────────────────────

def parse_binary_judge_output(judge_text: str) -> str:
    match = re.search(r'"verdict"\s*:\s*"([^"]+)"', str(judge_text))
    if match:
        val = match.group(1).strip().lower()
        if val in ("correct", "incorrect"):
            return val
    return "fail"

# ── Config ───────────────────────────────────────────────────────────────────

RESULTS_DIR = "/home/tmalik6/Summer/dedup/Summer_Code"

EXPERIMENTS = [
    {"key": "exp1", "label": "Llama-3.1-8B (self)", "gsm8k_csv": f"{RESULTS_DIR}/GSM8K_l8_binary_eval_exp1_llama8b.csv",   "math500_csv": f"{RESULTS_DIR}/MATH500_l8_binary_eval_exp1_llama8b.csv"},
    {"key": "exp2", "label": "Gemma-2-9B",          "gsm8k_csv": f"{RESULTS_DIR}/GSM8K_l8_binary_eval_exp2_gemma2_9b.csv", "math500_csv": f"{RESULTS_DIR}/MATH500_l8_binary_eval_exp2_gemma2_9b.csv"},
    {"key": "exp3", "label": "Llama-3.3-70B",       "gsm8k_csv": f"{RESULTS_DIR}/GSM8K_l8_binary_eval_exp3_llama70b.csv", "math500_csv": f"{RESULTS_DIR}/MATH500_l8_binary_eval_exp3_llama70b.csv"},
]

# Oracle condition: judge is shown the true/target solution (always correct by
# construction). Reuses the same Target CSVs as the original target-eval script.
ORACLE = {"key": "oracle", "label": "True Solution (oracle)", "gsm8k_csv": f"{RESULTS_DIR}/GSM8K_l8_binary_eval_target.csv", "math500_csv": f"{RESULTS_DIR}/MATH500_l8_binary_eval_target.csv"}

# ── Core analysis ────────────────────────────────────────────────────────────

def analyse_binary(dataset_name, label, solutions, judge_raws, gt_answers, extractor, comparator):
    l = len(solutions)

    actual_correct = 0
    actual_incorrect = 0

    verdict_correct = 0
    verdict_incorrect = 0
    parse_fail = 0

    # confusion counts vs ground truth
    tp = 0  # actually correct, judge said correct
    fn = 0  # actually correct, judge said incorrect
    fp = 0  # actually incorrect, judge said correct
    tn = 0  # actually incorrect, judge said incorrect

    agree = 0  # judge verdict matches ground truth (parse_fail excluded from numerator)

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

    print(f"\n{'═' * 60}")
    print(f"  {dataset_name}  —  Binary judge (Llama-3.1-8B local)  —  Generator: {label}")
    print(f"{'═' * 60}")
    print(f"  Total questions               : {l}")

    print(f"\n  ── Ground truth breakdown ─────────────────────────────")
    print(f"  Generator actually correct    : {actual_correct}  ({actual_correct/l*100:.1f}%)")
    print(f"  Generator actually incorrect  : {actual_incorrect}  ({actual_incorrect/l*100:.1f}%)")

    print(f"\n  ── Judge verdict distribution ─────────────────────────")
    print(f"  Verdict = correct             : {verdict_correct}  ({verdict_correct/l*100:.1f}%)")
    print(f"  Verdict = incorrect           : {verdict_incorrect}  ({verdict_incorrect/l*100:.1f}%)")
    print(f"  Parse fail                    : {parse_fail}  ({parse_fail/l*100:.1f}%)")

    print(f"\n  ── Confusion matrix (judge vs ground truth, parse-fail counted as wrong) ──")
    print(f"  TP (actual✓, judge✓)          : {tp}")
    print(f"  FN (actual✓, judge✗ or fail)  : {fn + parse_fail}")
    print(f"  FP (actual✗, judge✓)          : {fp}")
    print(f"  TN (actual✗, judge✗)          : {tn}")

    accuracy = agree / l if l > 0 else None
    print(f"\n  ── Judge accuracy (agreement with ground truth, parse-fail = wrong) ───")
    print(f"  Accuracy                      : {agree}/{l}  ({accuracy*100:.1f}%)")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision (verdict=correct)   : {tp}/{tp+fp}  ({precision*100:.1f}%)")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall (verdict=correct)      : {tp}/{tp+fn}  ({recall*100:.1f}%)")

    return {
        "dataset": dataset_name,
        "experiment": label,
        "n": l,
        "actual_correct": actual_correct,
        "verdict_correct": verdict_correct,
        "verdict_incorrect": verdict_incorrect,
        "parse_fail": parse_fail,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": accuracy,
    }

# ── Per-dataset wrappers ────────────────────────────────────────────────────

def analyse_gsm8k(csv_path, label, gt_answers):
    df = pd.read_csv(csv_path)
    solutions  = df["Solution"].to_list()
    judge_raws = df["Judge L8"].to_list()

    def extractor(raw):
        return extract_final_answer(raw)

    return analyse_binary(
        dataset_name="GSM8K",
        label=label,
        solutions=solutions,
        judge_raws=judge_raws,
        gt_answers=gt_answers,
        extractor=extractor,
        comparator=num_compare,
    )

def analyse_math500(csv_path, label, gt_answers):
    df = pd.read_csv(csv_path)
    solutions  = df["Solution"].to_list()
    judge_raws = df["Judge L8"].to_list()

    def extractor(raw):
        ans = extract_math_ans(str(raw))
        return "NA" if ans == "NA" else ans

    return analyse_binary(
        dataset_name="MATH500",
        label=label,
        solutions=solutions,
        judge_raws=judge_raws,
        gt_answers=gt_answers,
        extractor=extractor,
        comparator=math_compare,
    )

# Oracle wrappers: same analyse_binary call, but solutions come from the
# Target column (the gold solution itself) and the comparator is always_correct
# since the target is correct by construction. gt_answers is unused by
# always_correct, so we pass a dummy list to satisfy the shared interface.

def analyse_gsm8k_oracle(csv_path, label):
    df = pd.read_csv(csv_path)
    solutions  = df["Target"].to_list()
    judge_raws = df["Judge L8"].to_list()

    def extractor(raw):
        return raw  # unused by always_correct, kept for interface consistency

    return analyse_binary(
        dataset_name="GSM8K",
        label=label,
        solutions=solutions,
        judge_raws=judge_raws,
        gt_answers=[None] * len(solutions),
        extractor=extractor,
        comparator=always_correct,
    )

def analyse_math500_oracle(csv_path, label):
    df = pd.read_csv(csv_path)
    solutions  = df["Target"].to_list()
    judge_raws = df["Judge L8"].to_list()

    def extractor(raw):
        return raw  # unused by always_correct, kept for interface consistency

    return analyse_binary(
        dataset_name="MATH500",
        label=label,
        solutions=solutions,
        judge_raws=judge_raws,
        gt_answers=[None] * len(solutions),
        extractor=extractor,
        comparator=always_correct,
    )

# ── Run ───────────────────────────────────────────────────────────────────────

print("Loading GSM8K dataset for ground truth...")
gsm8k_ds = load_dataset("openai/gsm8k", "main")
gsm8k_gt = [extract_gsm8k_ans(a) for a in gsm8k_ds['test']['answer']]

print("Loading MATH-500 dataset for ground truth...")
math500_ds = load_dataset("HuggingFaceH4/MATH-500")
math500_gt = [extract_math_ans(str(s)) for s in math500_ds['test']['solution']]

all_results = []
for exp in EXPERIMENTS:
    all_results.append(analyse_gsm8k(exp["gsm8k_csv"], exp["label"], gsm8k_gt))
    all_results.append(analyse_math500(exp["math500_csv"], exp["label"], math500_gt))

all_results.append(analyse_gsm8k_oracle(ORACLE["gsm8k_csv"], ORACLE["label"]))
all_results.append(analyse_math500_oracle(ORACLE["math500_csv"], ORACLE["label"]))

# ── Side-by-side summary across all experiments ────────────────────────────────

print(f"\n{'═' * 60}")
print(f"  SUMMARY — Binary judge accuracy across experiments")
print(f"{'═' * 60}")
summary_df = pd.DataFrame(all_results)[["dataset", "experiment", "n", "actual_correct", "accuracy", "parse_fail"]]
summary_df["actual_correct_pct"] = (summary_df["actual_correct"] / summary_df["n"] * 100).round(1)
summary_df["accuracy_pct"] = (summary_df["accuracy"] * 100).round(1)
print(summary_df[["dataset", "experiment", "n", "actual_correct_pct", "accuracy_pct", "parse_fail"]].to_string(index=False))

summary_df.to_csv("binary_judge_summary.csv", index=False)
print("\nSummary saved to binary_judge_summary.csv")