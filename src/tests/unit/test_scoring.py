from mas_sae.evaluation.scoring import answer_matches, normalize_answer


def test_normalize_answer_handles_case_articles_and_punctuation():
    assert normalize_answer("The Miquette Giraudy!") == "miquette giraudy"


def test_answer_matches_gold():
    assert answer_matches("Miquette Giraudy", "Miquette Giraudy")


def test_answer_matches_alias():
    assert answer_matches(
        "U.S.",
        "United States",
        aliases=["US", "U.S."],
    )


def test_answer_matches_returns_false_for_wrong_answer():
    assert not answer_matches(
        "London",
        "Paris",
        aliases=["Paris, France"],
    )


def test_answer_matches_none_returns_false():
    assert not answer_matches(None, "Paris")
