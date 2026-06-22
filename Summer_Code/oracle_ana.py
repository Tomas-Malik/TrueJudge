import pandas as pd
import re
from typing import Optional
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

# ── Judge output parser ───────────────────────────────────────────────────────

def parse_judge_output(judge_text: str) -> str:
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(judge_text))
    if match:
        val = match.group(1).strip().lower()
        if val in ("solution_1", "solution_2", "none_correct", "both_correct"):
            return val
    return "fail"

# ── Core analysis ─────────────────────────────────────────────────────────────
# ── Global config ─────────────────────────────────────────────────────────────

TRUE_SOL_POSITION = "solution_2"  # set to "solution_1" for swapped-position CSVs

# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_2way(
    dataset_name: str,
    questions: list,
    g9_raws: list,
    target_raws: list,
    judge_raws: list,
    g9_extractor,
    target_extractor,
    comparator,
):
    l = len(questions)

    true_sol  = TRUE_SOL_POSITION                                        # e.g. "solution_2"
    gemma_sol = "solution_1" if true_sol == "solution_2" else "solution_2"

    gt_both_right = 0
    gt_g9_only    = 0

    picked_solution_1  = 0
    picked_solution_2  = 0
    picked_none        = 0
    picked_both        = 0
    parse_fail         = 0

    correct_when_picked_gemma    = 0   # gemma was actually right when judge picked gemma
    correct_when_picked_true_sol = 0   # gemma was actually WRONG when judge picked true sol
    correct_when_picked_none     = 0   # gemma actually wrong when judge said none correct
    correct_when_picked_both     = 0   # gemma actually right when judge said both correct

    scenario_both_right_total = 0; scenario_both_right_correct = 0
    scenario_g9_wrong_total   = 0; scenario_g9_wrong_correct   = 0

    method_correct = 0

    for q, g9_raw, target_raw, judge_raw in zip(questions, g9_raws, target_raws, judge_raws):
        g9_ans     = g9_extractor(g9_raw)
        target_ans = target_extractor(target_raw)
        g9_ans     = "NA_g9" if g9_ans == "NA" else g9_ans
        g9_correct = comparator(g9_ans, target_ans)

        if g9_correct:
            gt_both_right += 1
        else:
            gt_g9_only += 1

        decision = parse_judge_output(judge_raw)

        if g9_correct:
            scenario_both_right_total += 1
            if decision == "both_correct":
                scenario_both_right_correct += 1
        else:
            scenario_g9_wrong_total += 1
            if decision == true_sol:
                scenario_g9_wrong_correct += 1

        if decision == "fail":
            parse_fail += 1
            if g9_correct:
                method_correct += 1

        elif decision == "solution_1":
            picked_solution_1 += 1
            if decision == gemma_sol:
                if g9_correct:
                    correct_when_picked_gemma += 1
                    method_correct += 1
            else:  # decision == true_sol
                if not g9_correct:
                    correct_when_picked_true_sol += 1
                method_correct += 1

        elif decision == "solution_2":
            picked_solution_2 += 1
            if decision == gemma_sol:
                if g9_correct:
                    correct_when_picked_gemma += 1
                    method_correct += 1
            else:  # decision == true_sol
                if not g9_correct:
                    correct_when_picked_true_sol += 1
                method_correct += 1

        elif decision == "none_correct":
            picked_none += 1
            if not g9_correct:
                correct_when_picked_none += 1

        elif decision == "both_correct":
            picked_both += 1
            if g9_correct:
                correct_when_picked_both += 1
                method_correct += 1

    picked_gemma    = picked_solution_1 if gemma_sol == "solution_1" else picked_solution_2
    picked_true_sol = picked_solution_1 if true_sol  == "solution_1" else picked_solution_2

    print(f"\n{'═' * 60}")
    print(f"  {dataset_name}  —  2-way blind judge (Llama-3.1-8B local)")
    print(f"  [True solution presented as: {true_sol}]")
    print(f"{'═' * 60}")
    print(f"  Total questions              : {l}")

    print(f"\n  ── Ground truth breakdown ─────────────────────────────")
    print(f"  Gemma correct (both right)   : {gt_both_right}  ({gt_both_right/l*100:.1f}%)")
    print(f"  Gemma wrong (target right)   : {gt_g9_only}  ({gt_g9_only/l*100:.1f}%)")

    print(f"\n  ── Judge decision distribution ────────────────────────")
    print(f"  Picked solution_1 (Gemma)    : {picked_solution_1}  ({picked_solution_1/l*100:.1f}%)" if gemma_sol == "solution_1" else
          f"  Picked solution_1 (true sol) : {picked_solution_1}  ({picked_solution_1/l*100:.1f}%)")
    print(f"  Picked solution_2 (target)   : {picked_solution_2}  ({picked_solution_2/l*100:.1f}%)" if true_sol == "solution_2" else
          f"  Picked solution_2 (Gemma)    : {picked_solution_2}  ({picked_solution_2/l*100:.1f}%)")
    print(f"  Picked none_correct          : {picked_none}  ({picked_none/l*100:.1f}%)")
    print(f"  Picked both_correct          : {picked_both}  ({picked_both/l*100:.1f}%)")
    print(f"  Parse fail                   : {parse_fail}  ({parse_fail/l*100:.1f}%)")

    print(f"\n  ── Precision of each pick ─────────────────────────────")
    if picked_gemma > 0:
        print(f"  {gemma_sol} precision (Gemma)  : {correct_when_picked_gemma}/{picked_gemma}"
              f"  ({correct_when_picked_gemma/picked_gemma*100:.1f}%)"
              f"  ← Gemma was actually right")
    if picked_true_sol > 0:
        print(f"  {true_sol} precision (true) : {correct_when_picked_true_sol}/{picked_true_sol}"
              f"  ({correct_when_picked_true_sol/picked_true_sol*100:.1f}%)"
              f"  ← Gemma was actually wrong")
    if picked_none > 0:
        print(f"  none_correct precision       : {correct_when_picked_none}/{picked_none}"
              f"  ({correct_when_picked_none/picked_none*100:.1f}%)"
              f"  ← Gemma actually wrong")
    if picked_both > 0:
        print(f"  both_correct precision       : {correct_when_picked_both}/{picked_both}"
              f"  ({correct_when_picked_both/picked_both*100:.1f}%)"
              f"  ← Gemma actually right")

    print(f"\n  ── Judge scenario identification accuracy ─────────────")
    if scenario_both_right_total > 0:
        print(f"  When Gemma correct → picked both_correct  : "
              f"{scenario_both_right_correct}/{scenario_both_right_total}"
              f"  ({scenario_both_right_correct/scenario_both_right_total*100:.1f}%)")
    if scenario_g9_wrong_total > 0:
        print(f"  When Gemma wrong   → picked {true_sol}    : "
              f"{scenario_g9_wrong_correct}/{scenario_g9_wrong_total}"
              f"  ({scenario_g9_wrong_correct/scenario_g9_wrong_total*100:.1f}%)")

    print(f"\n  ── Method accuracy ────────────────────────────────────")
    print(f"  Method correct               : {method_correct}/{l}  ({method_correct/l*100:.1f}%)")
    print(f"  Gemma baseline               : {gt_both_right}/{l}  ({gt_both_right/l*100:.1f}%)")
    print(f"  Oracle upper bound           : {l}/{l}  (100.0%)  ← target always right")
# ── GSM8K ─────────────────────────────────────────────────────────────────────

def analyse_gsm8k():
    df = pd.read_csv("/home/tmalik6/Summer/dedup/Summer_Code/GSM8K_l8_judge_2way_blind_fullsoln.csv")

    questions   = df["Question"].to_list()
    g9_raws     = df["G9"].to_list()
    target_raws = df["Target"].to_list()
    judge_raws  = df["Judge L8"].to_list()

    # Load ground truth answers from HuggingFace (#### format)
    print("Loading GSM8K dataset for ground truth...")
    ds = load_dataset("openai/gsm8k", "main")
    gt_answers = [extract_gsm8k_ans(a) for a in ds['test']['answer']]

    # For GSM8K the "target" column is the full solution text;
    # extract the number from it as the reference for comparison
    def g9_extractor(raw):
        return extract_final_answer(raw)

    def target_extractor(raw):
        return extract_gsm8k_ans(raw)

    # Override target_raws with gt_answers (indexed) for comparator consistency
    # but we still pass target_raws to extract from — they come from the same source
    analyse_2way(
        dataset_name="GSM8K",
        questions=questions,
        g9_raws=g9_raws,
        target_raws=target_raws,
        judge_raws=judge_raws,
        g9_extractor=g9_extractor,
        target_extractor=target_extractor,
        comparator=num_compare,
    )

# ── MATH500 ───────────────────────────────────────────────────────────────────

def analyse_math500():
    df = pd.read_csv("/home/tmalik6/Summer/dedup/Summer_Code/MATH500_l8_judge_2way_blind_fullsoln.csv")

    questions   = df["Question"].to_list()
    g9_raws     = df["G9"].to_list()
    target_raws = df["Target"].to_list()
    judge_raws  = df["Judge L8"].to_list()

    def g9_extractor(raw):
        ans = extract_math_ans(str(raw))
        return "NA" if ans == "NA" else ans

    def target_extractor(raw):
        ans = extract_math_ans(str(raw))
        return "NA" if ans == "NA" else ans

    analyse_2way(
        dataset_name="MATH500",
        questions=questions,
        g9_raws=g9_raws,
        target_raws=target_raws,
        judge_raws=judge_raws,
        g9_extractor=g9_extractor,
        target_extractor=target_extractor,
        comparator=math_compare,
    )

# ── Run ───────────────────────────────────────────────────────────────────────

analyse_gsm8k()
analyse_math500()