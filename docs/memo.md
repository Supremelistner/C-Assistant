# Research memo: which LLM, and when a small model is better

**Student Success Assistant · 24 Oct 2026 · prices checked 22–23 Oct 2026 · numbers measured on our own data**

## Question A: Which LLM for grading 50,000 students a month?

**Recommendation: Qwen3.8-27B (open weights, Apache-2.0) with the `rag` prompt, about $87 a month.** Pilot it on
Groq with Zero Data Retention switched on and parental consent collected. Then move the *same* open model to
India-hosted servers before scaling to under-18s. **Gemma 4 31B is the tested fallback.**

**Assumptions.**

- **Volume:** 50,000 students × 1.45 explanations each (724 explanations ÷ 500 students in our data) =
  **72,500 gradings a month**. This is cautious: 1.45 is per course, not per month. High case: 150,000 (3 lessons).
- **Tokens** are measured from our runs: about 975 in and 105 out per answer with the `rag` prompt.
- **Retries:** none were needed in 189 Qwen calls, so they are ignored.

| | **Qwen3.8-27B · Groq** | Gemini 3 Flash · Google | Gemma 4 31B · DeepInfra |
|---|---|---|---|
| Price per 1M tokens, in / out | $0.80 / $4.00 [1] | $0.50 / $3.00 [2] | ~$0.13 / $0.38 [3]* |
| **Monthly cost, base (high)** | **$87 ($181)** | $58 ($120)† | $12 ($25) |
| Speed (our median per answer) | **8.5 s** (Groq states 450 tokens/s [1]) | 6.7 s | 17.4 s |
| Quality: 60 teacher labels | **98.3%** (59/60); 0 wrong answers passed; 0/4 injections worked | not run | not run |
| Quality: 10 random labels | 10/10 | 10/10 | 10/10 |
| Student-data privacy | Not retained by default; Zero Data Retention available; stored in the US [4] | Developer API **barred for apps used by under-18s** [5]; paid tier not used for training | Open weights: can run on India-hosted servers |
| Hindi / Tamil / Kannada | No languages listed on the model card | Hindi listed; Tamil and Kannada not on the API list [6] | "35+ supported, 140+ pre-trained"; no Indic scores [7] |

*Calculation, Qwen base case:*

- Input: 72,500 × 976 = 70.8M tokens × $0.80 per million = **$56.61**
- Output: 72,500 × 106 = 7.7M tokens × $4.00 per million = **$30.63**
- **Total: $87.24 a month** (about $1.20 per 1,000 answers)

The other columns use the same method with their own prices.

- \*Gemma's paid price comes from third-party trackers, which disagree ($0.13–0.27 for input).
- †Gemini bills its hidden "thinking" tokens as output [2]. Our token counts may not include them, so its true cost
  is probably higher.

**Why Qwen.** Even in the high case, every option costs under $190 a month, so cost doesn't decide this; quality and
data privacy do. In one line: **tested, fast and portable.**

- **Tested:** it's the only model evaluated on all 60 teacher labels: 86.7% → 96.7% → **98.3%** from prompt v1 to
  v2 to `rag`. Moving from v1 to `rag` fixed 7 answers and broke none (sign test p = 0.016).
- **Fast:** about 8.5 s per answer, against 17.4 s for Gemma.
- **Portable:** its weights are open, so it can move to another host, or to India, without changing the model.

**Why not the others.**

- **Gemini 3 Flash** is cheaper per token, but Google's developer terms bar apps likely to be used by under-18s. It
  would need Vertex AI on enterprise terms, and we couldn't confirm it runs in the Mumbai region. It's also a
  *Preview* model with newer stable versions available [8].
- **Gemma 4 31B** matched Qwen on the 10-answer sample and is about 7× cheaper, but twice as slow. Ten answers are
  too few to switch on, which makes it the fallback. Evaluate it on all 60 labels before relying on it.

**Risk 1: Qwen is a Preview model on Groq.** Groq says Preview models are "for evaluation, not production" [1], so
the price or the model itself can change at short notice.
*Mitigation:* the open weights run on other hosts. Switching is one line in `src/grader.py` (`MODELS`), and Gemma is
already tested.

**Risk 2: Children's data under India's DPDP law.** The DPDP Rules 2025 require **verifiable parental consent** to
process the data of anyone under 18 [9], and Groq stores data in the US [4].
*Mitigation:* collect consent at signup; send only the answer text and the lesson ID (no name or student ID); turn
on Zero Data Retention; move to India-hosted inference before enrolling minors at scale.

**What we could not confirm.**

- **Indian-language quality:** no provider publishes Hindi, Tamil or Kannada scores for a task like this. We tested
  only 3 Hinglish answers (`rag` got all 3 right), and no Tamil or Kannada script.
- **Gemini's thinking tokens:** how many it uses. 500 extra per answer would add about $109 a month.
- **Gemma's paid price,** and Groq's contractual no-training clause.
- **Gemini 3 Flash** availability in India.
- **The DPDP education exemption:** whether it covers private edtech.
- **The 10-answer sample** can't rank the models; it only shows that all three are viable.

## Question B: When should a company use a small ML model instead of an LLM?

**Our view.** **Use a small model** when:

- the answer is a fixed label or a number;
- labelled examples are plentiful, or the data comes in tidy columns;
- volume is high and every prediction must be cheap and instant;
- decisions have to be explained or audited.

**Use an LLM** when the input is open-ended language, labels are scarce, or the output is itself text, such as
feedback.

| | Small ML model | LLM |
|---|---|---|
| Cost | Close to $0 per prediction; runs on a CPU | ~$1.20 per 1,000 answers here; grows with tokens |
| Speed | Under 1 millisecond | Seconds (8.5 s here) |
| Accuracy | Strong when the training data covers the task | Strong on open text with few labels; can be inconsistent |
| Explainability | Coefficients and feature importance | Written reasons, not guaranteed to be the real reasons |
| Data needed | Hundreds to thousands of labelled examples | A rubric plus a handful of examples |
| Maintenance | Retrain when the data drifts; you own the model | Prompt changes, vendor versions, price changes, injection defences |

**Real-world examples.**

- **Gmail spam filtering:** Google reported that a TensorFlow model blocks about 100 million extra spam messages a
  day, on top of the 99.9% its existing ML already stopped [10].
- **Stripe Radar:** scores every card payment in real time from thousands of signals, and is retrained daily [11].

These are billions of yes/no decisions that need answers in milliseconds, with labels arriving for free (spam
reports, chargebacks). An LLM call per email or payment would be slower, costlier and harder to audit. The opposite
case is **Duolingo Max**, which uses GPT-4 to *explain* why a learner's answer is wrong and to role-play
conversations [12]. That's open-ended text with no fixed answer, which is LLM territory and close to our Part 2.

**Applied to this project.**

- **Part 1 (predicting dropouts): small model.** The data is 500 rows of tidy, labelled columns. Logistic regression
  reaches ROC-AUC 0.76, scores in milliseconds, gives probabilities we could calibrate a threshold on (0.35 catches
  81% of dropouts), and its coefficients explain each flag to a tutor. An LLM would add cost and opacity for no gain.
- **Part 2 (judging explanations): LLM.** The answers are free text, some in Hinglish, with only 60 labels, and the
  output must include written feedback and a hint.

**Bonus: TF-IDF + logistic regression on the 60 labels** (`src/baseline_text_clf.py`)

| Method | Agreement with teacher | Wrong answers passed | Cost / speed |
|---|---|---|---|
| TF-IDF + LR, leave-one-out | 80.0% (misleading) | 4 of 20 | ~$0 · 0.6 ms |
| TF-IDF + LR, near-copies held out together | **38.3%** (chance is 33%) | 13 of 20 | ~$0 · 0.6 ms |
| Qwen3.8-27B, `rag` prompt | **98.3%** | 0 of 20 | $1.20 per 1,000 · 8.5 s |

**Sixty rows is far too few.** They contain only 37 distinct answers; the rest are copies with a different hedge
("i think"). Under leave-one-out, the 38 rows with a near-copy in training scored 100%, against 45% for the rest:
the classifier memorises templates, it doesn't learn to grade. A random 80/20 split has the same leak (76%, against
43% when copies are kept together). A fair text classifier would need hundreds of labels, split by answer template.

**Sources** (accessed 22–24 Oct 2026):
[1] console.groq.com/docs/models ·
[2] ai.google.dev/gemini-api/docs/pricing ·
[3] computeprices.com and typingmind.com (Gemma 4 31B price trackers) ·
[4] console.groq.com/docs/your-data ·
[5] ai.google.dev/gemini-api/terms ·
[6] ai.google.dev/available_regions (supported languages) ·
[7] huggingface.co/google/gemma-4-31B-it ·
[8] ai.google.dev/gemini-api/docs/models ·
[9] MeitY DPDP Rules 2025, summarised by SCC Online (26 Dec 2025) ·
[10] 9to5google.com, 6 Feb 2019 (Google's announcement) ·
[11] support.stripe.com/topics/radar ·
[12] investors.duolingo.com, "Duolingo Max Shows the Future of AI Education", 14 Mar 2023.
Our measurements: `data/llm/graded.csv`, `data/llm/compare.csv`, `data/llm/baseline_text_clf.csv`.
