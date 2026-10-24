"""Find teacher-graded answers that look like the one being graded (the "R" in RAG).

Similarity is TF-IDF on character n-grams: robust to typos and Hinglish, runs locally, costs nothing.

Leak guard: the graded examples are also the answers we evaluate on, and many answers are the same text
with a different hedge on the end. Showing the model a near-copy of the answer *with its grade* would hand
it the answer key, so we never show the answer itself or anything more similar than NEAR_DUPLICATE.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEDGES = r"i think|not sure tho|hope this is right!?|\(sorry for typos\)"
NEAR_DUPLICATE = 0.85


def core_text(text):
    """The answer without the hedge on the end — many answers differ only by that."""
    return re.sub(HEDGES, "", text, flags=re.I).strip()


class ExampleRetriever:
    def __init__(self, graded):
        """graded: DataFrame of teacher-labelled answers (explanation_id, lesson_id, label, explanation_text)."""
        kb = graded.assign(core=graded["explanation_text"].map(core_text))
        self.kb = kb.drop_duplicates(["core", "label"]).reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.kb["core"])

    def retrieve(self, text, lesson_id, exclude_id=None):
        """The most similar example for each verdict (understood / partial / wrong) in the same lesson."""
        sims = cosine_similarity(self.vectorizer.transform([core_text(text)]), self.matrix)[0]
        allowed = ((self.kb["lesson_id"] == lesson_id) & (self.kb["explanation_id"] != exclude_id)
                   & (sims < NEAR_DUPLICATE)).to_numpy()
        examples = []
        for label in ("understood", "partial", "wrong"):
            idx = np.flatnonzero(allowed & (self.kb["label"] == label).to_numpy())
            if len(idx):
                best = idx[np.argmax(sims[idx])]
                examples.append({"explanation_id": self.kb.at[best, "explanation_id"], "label": label,
                                 "text": self.kb.at[best, "core"], "similarity": round(float(sims[best]), 3)})
        return examples
