"""Related filler paragraphs: the pool paragraphs most similar to a question and its evidence (#156).

Applied to finished synthetic records by ``scripts/experiments/swap_synthetic_fillers.py``; the
generator itself is unchanged and keeps drawing random fillers.
"""
import random

from mas_sae.evaluation.scoring import normalize_answer


class RelatedFillers:
    """A TF-IDF index over filler pool paragraphs, built once and reused for every question."""

    def __init__(self, paragraphs: list[dict], *, top: int = 20):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.paragraphs = paragraphs
        self.top = top
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(f"{p['title']} {p['paragraph_text']}" for p in paragraphs)

    def draw(self, k: int, rng: random.Random, *, query: str, exclude_titles: set[str] = frozenset(),
             exclude_mentions: list[str] = ()) -> list[dict]:
        """Draw k of the ``top`` eligible paragraphs most similar to ``query``.

        A paragraph is ineligible if its title is excluded or its title or text mentions any excluded
        answer or name (after ``normalize_answer``), so a related filler cannot restate or contradict a
        hop's answer or name the question's own entities. Ties keep pool order, and the draw uses ``rng``,
        so the same query and seed give the same fillers.
        """
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        needles = [f" {n} " for n in {normalize_answer(m) for m in exclude_mentions} if n]
        eligible = []
        for i in sorted(range(len(self.paragraphs)), key=lambda i: -scores[i]):
            if len(eligible) == max(self.top, k):
                break
            paragraph = self.paragraphs[i]
            text = f" {normalize_answer(paragraph['title'] + ' ' + paragraph['paragraph_text'])} "
            if paragraph["title"] in exclude_titles or any(n in text for n in needles):
                continue
            eligible.append(paragraph)
        if len(eligible) < k:
            raise ValueError(f"Paragraph pool too small: needed {k}, found {len(eligible)}")
        return [dict(p) for p in rng.sample(eligible, k)]
