# Archive (not part of the current workflow)

These files are kept for transparency only. **No code reads them**, and nothing in the README's run steps uses them.

## `evidence_check_first_attempt/`

The first live run of the **evidence check** (24 Oct 2026, ~17:10–17:35). The check makes the model quote the
student's words for every key point it credits. **This run is incomplete and is not used for any reported number.**

| File | What it is |
|---|---|
| `graded.csv` | Qwen × 3 prompts × 64 answers: **134 of 189 ended in errors** |
| `at_risk_feedback.json` | The 5 at-risk results from this run |
| `run.log` | The run's log |

**Why it failed:**
- **71 errors: Groq's free daily limit** (200,000 tokens per day for Qwen3.8-27B), used up by the day's earlier runs.
- **63 errors (all `v1`): a bug, now fixed.** The naive `v1` prompt didn't describe the reply's fields, so when the
  strict-schema first try failed, the retry produced the wrong fields. `v1` now includes the schema.

**What it showed:** on the 55 answers that completed, Qwen credited 96 key points and **every quote was genuine**.
The full evaluation is pending; see `docs/grader.md`.

## `before_refactor/`

The grader's outputs from **23 Oct 2026**, before the final simplification (24 Oct). That run used the earlier
single-file `grader.py`, a cache keyed differently, and only partial runs of Gemini and Gemma (free-tier limits).

| File | What it was |
|---|---|
| `graded.csv` | Every grading result from the old run (Qwen × 3 prompts; Gemini 23 rows and Gemma 45 rows, prompt v1 only) |
| `cache.jsonl` | The old LLM reply cache. Its keys don't match the current grader, so it can't be reused |
| `at_risk_feedback.csv`, `at_risk_feedback.json` | The old feedback for the 5 at-risk students |
| `part2_llm_grader.md` | The old Part 2 write-up, replaced by `docs/grader.md` |
| `run_*.log`, `run_*.err` | Logs of the old grading runs |

**Why the numbers differ from today's.** The old run measured Qwen at 86.7% / 90.0% / 96.7% (prompt v1 / v2 / rag);
the 24 Oct run measured 86.7% / 96.7% / 98.3%. Same prompts, same data. LLM replies vary slightly between runs even at
temperature 0. The numbers in `data/llm/` and `docs/` are the ones to quote.
