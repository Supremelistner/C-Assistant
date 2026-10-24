# The LLM grader (Part 2): how it works and how well

Run date: **24 Oct 2026**. Every number here comes from `data/llm/` and is reproduced in
`src/evaluate_grader.ipynb`.

> **Two versions of the grader are described here.** The evaluated results were produced *before* the **evidence
> check** was added. The code now includes it: the model must quote the student's words for each key point it
> credits. That check is tested offline but **not yet evaluated** on the full set (see
> [The evidence check](#the-evidence-check-in-the-code-not-yet-evaluated)). Everything else is identical.

## What it does

The grader reads a student's explanation of a lesson and returns one JSON object: a verdict (`understood`,
`partial` or `wrong`), which of the lesson's 4 key points the student got right or missed, feedback, and a hint.

```json
{
  "explanation_id": "E0508",
  "lesson_id": "L1",
  "verdict": "partial",
  "got_right": ["Plants use light energy (captured by chlorophyll) to make food.",
                "The outputs are glucose and oxygen."],
  "missed": ["The inputs are carbon dioxide and water.",
             "It happens mainly in the leaves (in chloroplasts)."],
  "hint": "Besides sunlight, what two other things does a plant need to take in to make its food?",
  "feedback": "You correctly identified that plants use sunlight to make food and release oxygen. ...",
  "misconception": null,
  "instruction_in_text": false
}
```

## How it works

Three files, each with one job:

| File | Job |
|---|---|
| `src/prompts.py` | **What we ask and what we accept.** The rubric, the 3 prompt versions, and the two schemas (`Grade` for the LLM's reply, `GradeResult` for the output) |
| `src/retriever.py` | **Finding examples.** Picks 3 teacher-graded answers similar to the one being graded (the "R" in RAG) |
| `src/grader.py` | **The pipeline.** Data loading, the LLM connections, the 4-step graph, the cache and the command line |

Every explanation goes through four steps, built with LangGraph:

```
guard ──> retrieve ──> grade ──> validate ──> result
                         ^          │
                         └─ retry ──┘   (once, if the reply is broken or the API call fails)
```

1. **guard:** a regex looks for text that tries to instruct the grader, e.g. "ignore previous instructions" or
   "system note". If it matches, it sets a flag and warns the model.
2. **retrieve:** builds the prompt. For `rag`, it adds 3 teacher-graded example answers from the same lesson, one per
   verdict.
3. **grade:** sends the prompt to the LLM with a strict JSON schema, or reuses a saved reply if this exact prompt was
   graded before.
4. **validate:** checks the reply against `Grade`, then builds the `GradeResult`. `got_right` and `missed` are
   filled in code from the key-point numbers the model chose, using the rubric's own wording. So they never
   overlap, and a reply naming a key point that doesn't exist is rejected. *In the current code this step also runs
   the evidence check (below).*

**The three prompt versions:**
- **`v1`:** the rubric + "reply in JSON". The naive first attempt, kept as a baseline.
- **`v2`:** the student's text is fenced off and declared to be data, not instructions; explicit rules for each
  verdict; *judge the content, not the confidence* (hedges like "i think" are not errors); any language.
- **`rag`:** `v2` + 3 teacher-graded examples. The rubric says *what* counts; the examples show *where the line is*
  between verdicts.

**No peeking at the answer key.** The examples come from the same 60 labelled answers we evaluate on, and many
answers are the same text with a different hedge. The retriever never shows the answer being graded, or anything
more than 85% similar to it. Checked: no answer was ever shown itself or an identical copy; the most similar example
shown was 43% similar.

**Cache.** Each reply is saved under a hash of the model plus the exact prompt. Re-runs cost nothing, and changing a
prompt automatically means a fresh call. Only valid replies are saved.

## How it was evaluated

| Question | Data | Model(s) |
|---|---|---|
| Which prompt works best? | All 60 teacher labels + 4 prompt-injection attempts | Qwen3.8-27B (Groq), each prompt version |
| Which model works best? | 10 random labelled answers (seed 42: 5 wrong, 3 understood, 2 partial) | Qwen, Gemini 3 Flash, Gemma 4 31B, `rag` prompt |

## Results

**Prompt versions (Qwen, 60 labels):**

| Prompt | Agreement with teacher | 95% interval | Correct answers marked `partial` | Wrong answers passed |
|---|---|---|---|---|
| v1 | 86.7% | 78–95% | 8 (3 of them hedged) | 0 |
| v2 | 96.7% | 92–100% | 0 | 0 |
| **rag** | **98.3%** | 95–100% | 1 | 0 |

- **v1's only problem was strictness.** All 8 of its mistakes were good answers marked `partial`; three were just
  hedged ("not sure tho").
- **v1 → rag fixed 7 answers and broke none** (sign test p = 0.016). v2 → rag fixed 2 (E0255 and E0336, which v2 had
  marked too generously) and broke 1 (E0250).
- **No prompt ever passed a wrong answer,** including confidently wrong ones.
- **Hinglish:** only `rag` got all 3 Hinglish answers right. That's too few to generalise from.
- **Prompt injection:** all 4 attempts were graded `wrong` by every prompt. The guard flagged all 4; the model
  flagged 3 of 4 with `v1` and all 4 with `v2` and `rag`.
- **The one remaining `rag` miss:** E0250 (the bus-braking example, teacher: `understood`) was marked `partial` for
  not saying "an object at rest stays at rest" explicitly. That's strict, but defensible.
- **Reliability:** 0 errors and 0 retries in 189 calls.

**Model comparison (10 random labels, `rag`):**

| Model | Correct | Median time per answer | Tokens in / out |
|---|---|---|---|
| Qwen3.8-27B · Groq | 10/10 | 8.5 s | 980 / 103 |
| Gemini 3 Flash · Google | 10/10 | 6.7 s | 966 / 106 |
| Gemma 4 31B · Google | 10/10 | 17.4 s | 966 / 101 |

All three agreed on every answer. **Ten answers can't separate them**; this only shows all three are viable.
Gemini needed a larger reply budget: with 1,000 tokens, one reply was cut off mid-JSON, because its hidden
"thinking" shares the budget.

## Feedback for 5 at-risk students (Qwen, `rag`)

These are students the dropout model flags (risk ≥ 0.35). One explanation each, covering all three lessons, plus one
Hinglish answer. Saved as `data/llm/at_risk_feedback.json`.

| Student | Answer | Verdict | Key points right |
|---|---|---|---|
| S035, E0440 | Hinglish, Newton's first law | `partial` | 2 of 4 |
| S243, E0508 | "Plants use sunlight to make food and they give out oxygen" | `partial` | 2 of 4 |
| S112, E0470 | "Things keep moving until something stops them… inertia" | `partial` | 2 of 4 |
| S184, E0507 | Mangoes and supply/demand, hedged | `understood` | 3 of 4 |
| S112, E0247 | "Plants eat sunlight directly, sunlight is their food" | `wrong` | 0 of 4 |

Every hint is a question, and none states the answer outright.

## The evidence check (in the code, not yet evaluated)

**Why it exists.** Reviewing the 5 at-risk outputs showed that the key-point labels could be loose. E0508 got credit
for chlorophyll and glucose it never mentioned, and E0470's `got_right` contradicted its feedback.

**What it does** (`src/prompts.py`, `to_result()`):
- **The model must quote.** For every key point it credits, it must quote the student's exact words
  (`credited: [{key_point, quote}]`).
- **Our code checks each quote** really appears in the answer (`quote_in_text()`, which ignores case and spacing).
  Credit without a real quote is **rejected**, and the point moves to `missed`.
- **An unsupported `understood` is retried:** if fewer than 3 backed points remain, the whole reply is rejected and
  the grader asks again.
- **The output gains an `evidence` field,** pairing each `got_right` point with the student's quote. The app shows
  these quotes.

**What it proves, and what it doesn't.** It proves the quoted words *exist* in the answer, not that they *mean* the
key point. So it stops invented credit, but not generous interpretation. On E0508, Qwen quoted *"use sunlight to make
food"* for the chlorophyll point: a real quote, so the credit is kept.

**Status.**
- **Tested offline:** `python tests/test_evidence_check.py` covers real and invented quotes, spacing and case, an
  unsupported `understood`, a non-existent key point, and Hinglish.
- **First live run (24 Oct): stopped by Groq's free daily limit** of 200,000 tokens. The day's earlier runs had
  used most of it, and a full evaluation needs about 230,000. On the 55 answers it completed, Qwen credited 96 key
  points and **every quote was genuine** (0 rejected). The run is archived in `archive/evidence_check_first_attempt/`.
- **A bug found and fixed in that run:** the naive `v1` prompt didn't describe the fields, so when the strict-schema
  first try failed, the retry produced the wrong fields. `v1` now includes the schema, like `v2` and `rag`.

**To evaluate it (e.g. on demo day),** run from the project root:
```bash
python tests/test_evidence_check.py      # offline, instant
python src/grader.py one E0508            # one live answer: shows the quotes in "evidence"
python src/grader.py eval                 # the full set; then re-run src/evaluate_grader.ipynb
```

**Fallbacks, if the free quota runs out again:**
1. **Run only the deployed prompt.** About 64 calls and 90,000 tokens, which fits a fresh day's quota. Change
   `for v in VERSIONS` to `for v in ["rag"]` in the `eval` task.
2. **Upgrade Groq to its paid tier.** A full run costs about $0.30 at the prices in the memo.
3. **Use Gemma on Google** (`--model gemma`), the tested fallback model, which has a separate quota.
4. **For a live demo with no quota left,** use the app's **Mock** model, or keep "Reuse a saved reply" ticked for
   answers already graded.

## Known limits

- **Small evaluation:** 60 labels, only 3 of them Hinglish, and no Tamil or Kannada.
- **`got_right` depends on the model's key-point numbers.** For `v2` and `rag` these line up with the verdicts (every
  `understood` has at least 3 points). `v1` often listed none (20 `partial` replies with 0 points), so v1's lists
  aren't reliable.
- **Free tiers:** Gemini 3 Flash allows 20 requests a day. Groq allows 8,000 tokens a minute, and counts the
  *requested* `max_tokens`, which is why Qwen's budget is kept at 1,000; it also allows 200,000 tokens a day for this
  model.
