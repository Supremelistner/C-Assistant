# Student Success Assistant: Dataset

Synthetic data for an online course with 3 lessons and 500 students. Read the task brief for what to build.

| File | Rows | Columns |
|---|---|---|
| `students.csv` | 500 | student_id, age_group, country, signup_date, device, refund_requested |
| `quiz_attempts.csv` | ~2,450 | attempt_id, student_id, lesson_id, score (0–100), time_spent_min, attempted_at |
| `explanations.csv` | ~720 | explanation_id, student_id, lesson_id, explanation_text, label |
| `outcome.csv` | 500 | student_id, dropped_out (yes/no) |
| `lessons.md` | 3 lessons | summary and key points for each lesson |

Notes:
- The course ran from June to 30 September 2026.
- `label` in `explanations.csv` is one of `understood`, `partial`, `wrong`. Only 60 rows are labelled; the rest are blank.
- This is real-world-style data: check it before you trust it.
