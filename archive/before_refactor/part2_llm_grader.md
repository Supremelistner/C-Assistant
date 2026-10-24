# Part 2 — LLM grader: method and results

Run date: **23 Oct 2026**. Numbers come from `data/llm/graded.csv` and are reproduced in
[`src/evaluate_grader.ipynb`](../src/evaluate_grader.ipynb).

## What it does

`src/grader.py` grades a student's explanation of a lesson as `understood` / `partial` / `wrong`,
and writes feedback plus a hint, using the lesson's four key points as the rubric. It is a LangGraph pipeline:

```
guard ──> retrieve ──> grade ──> validate ──> END
                         ^          │
                         └─ retry ──┘   (once, if the reply isn't valid JSON)
```

| Node | Job |
|---|---|
| guard | Regex pre-check for text that tries to instruct the grader (prompt injection) |
| retrieve | Lesson rubric, plus (for `rag`) 3 teacher-graded examples from the same lesson, one per verdict, picked by TF-IDF similarity |
| grade | LLM call through LangChain, with a strict JSON schema |
| validate | Pydantic check of the reply. One retry, then `error` |

**Output: one JSON object per explanation, with a strict schema (`GradeResult`).** Every field is required, and
extra fields are rejected:

```json
{
  "explanation_id": "E0440",
  "lesson_id": "L2",
  "verdict": "partial",
  "got_right": ["An object at rest stays at rest; a moving object keeps moving at constant speed in a straight line.",
                "This tendency is called inertia."],
  "missed": ["This continues unless an unbalanced (external) force acts on it.",
             "More mass means more inertia."],
  "hint": "If a hockey puck is sliding on perfectly smooth ice with no friction or air resistance, what will happen ...?",
  "feedback": "You correctly identified that objects resist changes in motion ...",
  "misconception": null,
  "instruction_in_text": false
}
```

There are two schemas, with separate jobs:
- **`Grade`** is what the LLM must reply with. It returns the *numbers* of the key points it judged correct
  (`key_points_correct`).
- **`GradeResult`** is the public output. `got_right` and `missed` are filled in code from those numbers, using the
  rubric's exact wording. So they always quote the rubric, never overlap, and together cover all 4 key points. A
  reply citing a key point that doesn't exist (e.g. `5`) is rejected and retried.

Print both schemas with `python src/grader.py --schema`. Grade one answer with
`python src/grader.py --explanation E0440 --models qwen --versions rag`.

**Three prompt versions:**
- **v1:** rubric + "reply in JSON"
- **v2:** student text fenced as data; judge content, not confidence; explicit verdict thresholds; any language; the hint must not give the answer
- **rag:** v2 + the retrieved examples

**Retrieval leak guard:** the examples come from the same 60 labelled rows used for evaluation, and 23 of those
60 have a near-copy elsewhere (templated answers with different hedges). The retriever never shows the row being
graded, or anything with similarity ≥ 0.85, so the model never sees its own answer with its label.
Checked: 0 leaks; the most similar example shown had similarity 0.43.

## Models and coverage

| Model | Provider | Graded | Why not more |
|---|---|---|---|
| Qwen3.8-27B | Groq | v1, v2, rag × 63 rows (complete) | — |
| Gemini 3 Flash (preview) | Google Gemini API | v1, 23 rows | Free tier allows 20 requests/day for this model |
| Gemma 4 31B | Google Gemini API | v1, 45 rows | Stopped early to use cached results (~29 s per call) |

GLM-5.3-Flash, the original third choice, is not served by Groq or Google, and the Hugging Face account had
no Inference Providers credit. Rows that were never graded are marked `not_run` and **excluded**, not counted as wrong.

## Results

**Qwen, all 60 labels:**

| Prompt | Accuracy | 95% CI (bootstrap) | Hedged correct answers marked `partial` |
|---|---|---|---|
| v1 | 86.7% | 78–95% | 3 |
| v2 | 90.0% | 82–97% | 1 |
| rag | **96.7%** | 92–100% | 0 |

**Row by row:**

| Change | Rows fixed | Rows broken | Sign test p |
|---|---|---|---|
| v1 → v2 | 5 | 3 | 0.73 |
| v2 → rag | 5 | 1 | 0.22 |
| v1 → rag | 7 | 1 | 0.07 |

**Model comparison** (prompt v1, the 23 rows all three graded; indicative only):

| Model | Correct |
|---|---|
| Gemma 4 31B | 23/23 (also 45/45 on all its rows) |
| Gemini 3 Flash | 20/23 |
| Qwen3.8-27B | 18/23 |

## What the results say

1. **The prompt mattered more than anything else we could measure.** All 8 of v1's mistakes were correct answers
   graded too harshly as `partial`, 3 of them only because they were hedged. "Judge content, not confidence" (v2)
   and the graded examples (rag) removed that.
2. **No grader passed a `wrong` answer.** Recall on `wrong` was 100% for every model and prompt, so confidently
   wrong answers ("Any economist will tell you…") didn't fool them.
3. **Retrieval helped on ambiguous cases.** v2 over-corrected on two Hinglish `partial` answers; rag got all three
   Hinglish rows right. Only 3 labelled Hinglish rows exist, so this needs more labels.
4. **Injection: no hijacks.** All 4 probes were graded `wrong` by every model/prompt that saw them, the guard and
   the model's own flag both caught them, and the system-prompt probe leaked no prompt text.
5. **Gemma looks strongest, but the evidence is thin.** It was perfect on v1, but a 3–5 answer gap on 23 rows sits
   inside the uncertainty, and Gemma hasn't been run with v2/rag.

## Cost and speed (measured)

| | Prompt tokens / call | Completion tokens / call | Median latency | p90 latency |
|---|---|---|---|---|
| Qwen v1 | 176 | 114 | 6.4 s | 22.4 s |
| Qwen v2 | 787 | 106 | 5.6 s | 8.5 s |
| Qwen rag | 950 | 106 | 7.6 s | 8.6 s |
| Gemini v1 | 160 | 95 | 8.2 s | 22.3 s |
| Gemma v1 | 160 | 87 | 29.4 s | 69.0 s |

rag costs about 20% more prompt tokens than v2. Latency is on free tiers with client-side rate limiting, so it
overstates paid-tier latency.

## Feedback for 5 at-risk students (Qwen, rag)

| Student | Answer | Verdict |
|---|---|---|
| S112, E0247 | "Plants eat sunlight directly, sunlight is their food" | `wrong` |
| S035, E0440 | Hinglish, Newton's first law | `partial` |
| S112, E0470 | "Things keep moving until something stops them… inertia" | `partial` |
| S184, E0507 | Mangoes / supply and demand, hedged | `understood` |
| S243, E0508 | "Plants use sunlight to make food and they give out oxygen" | `partial` |

- All five hints are questions, and none states the answer outright.
- The feedback names the specific missing key point each time.
- Two outputs are worth reading critically:
  - **E0247:** the feedback ends with a question, so the separate hint is partly redundant.
  - **E0507:** marked `understood`, yet the feedback still points to a missing key point (price as a signal). That fits the rubric, which needs 3 of 4 points.

Full outputs are in `data/llm/at_risk_feedback.json` (one `GradeResult` per student), `at_risk_feedback.csv`, and
section 11 of the notebook. The per-output comments in the
notebook are left for the author to write.

## Known issues and next steps

- **Two disagreements remain for Qwen rag.**
  - **E0250:** marked `partial` for not stating "at rest stays at rest" explicitly. Arguably defensible.
  - **E0336** (*"leaves use sunlight and water to make sugar"*): marked `understood` with all 4 key points credited,
    while its own feedback says carbon dioxide and oxygen are missing. The model over-credited. A count rule
    wouldn't catch this, because it claimed 4 points. The fix is to have the LLM also list the missed points, and
    reject replies where the two lists disagree.
- **`got_right` / `missed` are only as good as the key-point numbers.** For v2 and rag they line up with the
  verdicts: no `understood` has fewer than 3 points, and no `wrong` has 3 or more. **v1 often returned an empty list**
  (20 `partial` replies with 0 points), so v1's `got_right` isn't reliable.
- **The evaluation is small:** 60 labels, 3 of them Hinglish. Most gains are directional, not statistically settled.
- **To finish the model comparison,** run `python src/grader.py --models gemma` and `--models gemini` when quota allows. Both resume from cache.
- **Qwen v1 had one transient format failure** (missing fields). It graded correctly on rerun.
