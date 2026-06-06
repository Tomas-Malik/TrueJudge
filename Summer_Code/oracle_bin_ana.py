import pandas as pd
import re
from grading import grader
from datasets import load_dataset

# ── Extractors ────────────────────────────────────────────────────────────────

gsm8k_fp = "/home/tmalik6/Summer/judge/Summer_Code/GSM8K_l8_binary_eval_target.csv"
math500_fp = "/home/tmalik6/Summer/judge/Summer_Code/MATH500_l8_binary_eval_target.csv"

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

# ── Judge output parser ───────────────────────────────────────────────────────

def parse_verdict(judge_text: str) -> str:
    match = re.search(r'"verdict"\s*:\s*"([^"]+)"', str(judge_text))
    if match:
        val = match.group(1).strip().lower()
        if val in ("correct", "incorrect"):
            return val
    return "fail"

# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_binary(
    dataset_name: str,
    questions: list,
    target_raws: list,
    judge_raws: list,
    target_extractor,   # fn(raw) -> extracted answer string/number
    is_correct_fn,      # fn(extracted) -> bool  (target is always correct)
):
    l = len(questions)

    verdict_correct   = 0   # judge said "correct"   (true positive)
    verdict_incorrect = 0   # judge said "incorrect"  (false negative)
    verdict_fail      = 0   # parse failed

    extract_fail = 0        # target extraction returned NA

    for q, target_raw, judge_raw in zip(questions, target_raws, judge_raws):
        target_ans = target_extractor(target_raw)
        if target_ans == "NA":
            extract_fail += 1

        verdict = parse_verdict(judge_raw)

        if verdict == "fail":
            verdict_fail += 1
        elif verdict == "correct":
            verdict_correct += 1
        elif verdict == "incorrect":
            verdict_incorrect += 1

    tp  = verdict_correct    # judge correctly identifies correct solution
    fn  = verdict_incorrect  # judge incorrectly rejects correct solution
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0   # false negative rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # true positive rate / recall

    print(f"\n{'═' * 60}")
    print(f"  {dataset_name}  —  binary eval on target solution (Llama-3.1-8B)")
    print(f"{'═' * 60}")
    print(f"  Total questions              : {l}")
    print(f"  Target extraction failures   : {extract_fail}  ({extract_fail/l*100:.1f}%)")

    print(f"\n  ── Verdict distribution ───────────────────────────────")
    print(f"  Judged correct   (TP)        : {verdict_correct}  ({verdict_correct/l*100:.1f}%)")
    print(f"  Judged incorrect (FN)        : {verdict_incorrect}  ({verdict_incorrect/l*100:.1f}%)")
    print(f"  Parse fail                   : {verdict_fail}  ({verdict_fail/l*100:.1f}%)")

    print(f"\n  ── Reliability metrics ────────────────────────────────")
    print(f"  True positive rate (recall)  : {tpr*100:.1f}%"
          f"  ← how often judge correctly accepts true solution")
    print(f"  False negative rate          : {fnr*100:.1f}%"
          f"  ← how often judge wrongly rejects true solution")
    print(f"\n  NOTE: since ALL solutions here are correct by construction,")
    print(f"  any 'incorrect' verdict is a judge error (false negative).")

# ── GSM8K ─────────────────────────────────────────────────────────────────────

def analyse_gsm8k_binary():
    df = pd.read_csv(gsm8k_fp)

    questions   = df["Question"].to_list()
    target_raws = df["Target"].to_list()
    judge_raws  = df["Judge L8"].to_list()

    def target_extractor(raw):
        return extract_gsm8k_ans(raw)

    analyse_binary(
        dataset_name="GSM8K",
        questions=questions,
        target_raws=target_raws,
        judge_raws=judge_raws,
        target_extractor=target_extractor,
        is_correct_fn=lambda x: x != "NA",
    )

# ── MATH500 ───────────────────────────────────────────────────────────────────

def analyse_math500_binary():
    df = pd.read_csv(math500_fp)

    questions   = df["Question"].to_list()
    target_raws = df["Target"].to_list()
    judge_raws  = df["Judge L8"].to_list()

    def target_extractor(raw):
        return extract_math_ans(str(raw))

    analyse_binary(
        dataset_name="MATH500",
        questions=questions,
        target_raws=target_raws,
        judge_raws=judge_raws,
        target_extractor=target_extractor,
        is_correct_fn=lambda x: x != "NA",
    )

# ── Run ───────────────────────────────────────────────────────────────────────

analyse_gsm8k_binary()
analyse_math500_binary()