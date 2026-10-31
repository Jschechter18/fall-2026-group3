"""Related fillers (#156): the pool paragraphs most similar to the question, never naming its answers or entities."""
import random

import pytest

from mas_sae.data.synthetic.related_fillers import RelatedFillers
from test_synthetic_generation import RAW_2HOP


def text(title, body):
    return {"title": title, "paragraph_text": body}


TOPICAL = [text(f"Observatory {i}", f"The observatory {i} houses a refracting telescope founded by astronomers "
                                    f"in a hill town; its dome and telescope survey comets.") for i in range(30)]
UNRELATED = [text(f"Album {i}", f"Album {i} is a rock record with guitar songs released by a band on tour.")
             for i in range(30)]
QUERY = " ".join([RAW_2HOP["question"], *(p["paragraph_text"] for p in RAW_2HOP["supporting_paragraphs"])])
ANSWERS = [RAW_2HOP["answer"], *(step["answer"] for step in RAW_2HOP["question_decomposition"])]


def test_related_fillers_prefer_paragraphs_about_the_question():
    fillers = RelatedFillers(UNRELATED + TOPICAL).draw(16, random.Random(0), query=QUERY)
    assert all(p["title"].startswith("Observatory") for p in fillers)


@pytest.mark.parametrize("title, body", [
    ("Observatory leak", "The observatory telescope dome survey founded astronomers. Selin was its earlier name."),
    ("Observatory leak", "The observatory telescope dome survey founded astronomers. It opened in 1842."),
    ("Observatory leak", "The observatory telescope dome survey founded astronomers. Its site, SELIN, is old."),
    ("Selin Observatory", "The observatory telescope dome survey founded astronomers."),   # the title alone
])
def test_related_fillers_never_mention_a_hop_answer(title, body):
    """A topical paragraph naming any step answer (normalized) could contradict the evidence.

    ``top`` equals k, so every eligible paragraph is drawn: the tainted one is absent only if excluded.
    """
    pool = RelatedFillers([text(title, body)] + TOPICAL + UNRELATED, top=16)
    fillers = pool.draw(16, random.Random(0), query=QUERY, exclude_mentions=ANSWERS)
    assert title not in [p["title"] for p in fillers]


def test_related_fillers_never_name_the_questions_own_entities():
    # As a real college "in Halden" was drawn for a question whose evidence is titled "Halden".
    named = text("Observatory college", "The observatory telescope dome survey founded near Luma Observatory.")
    pool = RelatedFillers([named] + TOPICAL + UNRELATED, top=16)
    fillers = pool.draw(16, random.Random(0), query=QUERY, exclude_titles={"Luma Observatory"},
                        exclude_mentions=[*ANSWERS, "Luma Observatory"])
    assert "Observatory college" not in [p["title"] for p in fillers]


def test_related_fillers_skip_excluded_titles_and_repeat_for_the_same_seed():
    pool = RelatedFillers(UNRELATED + TOPICAL, top=25)
    draw = lambda seed: pool.draw(16, random.Random(seed), query=QUERY, exclude_titles={"Observatory 0"})
    assert draw(3) == draw(3) and "Observatory 0" not in [p["title"] for p in draw(3)]


def test_small_related_pool_is_refused():
    with pytest.raises(ValueError, match="pool too small"):
        RelatedFillers(TOPICAL[:5]).draw(16, random.Random(0), query=QUERY)
