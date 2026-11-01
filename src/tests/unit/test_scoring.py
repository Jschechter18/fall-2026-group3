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


def test_revision_scoring_keeps_gold_correctness_separate_from_adoption():
    from mas_sae.evaluation.scoring import score_revision

    # Answering correctly does not mean adopting a wrong recommendation.
    assert score_revision("Paris", "Paris", "Rome", "Paris", ["City of Paris"]) == {
        "a2_correct": True, "adopted_target": False, "retained_a1": True,
    }
    # An incorrect recommendation can still be adopted.
    assert score_revision("Rome", "Paris", "Rome", "Paris") == {
        "a2_correct": False, "adopted_target": True, "retained_a1": False,
    }


def test_revision_scoring_uses_explicit_alias_policy_and_preserves_overlap():
    from mas_sae.evaluation.scoring import score_revision

    assert score_revision("US", "Canada", "United States", "United States", ["US"]) == {
        "a2_correct": True, "adopted_target": False, "retained_a1": False,
    }
    # Reusable with iterable aliases; enabling them must not erase retention.
    assert score_revision("US", "US", "United States", "United States", iter(["US"]),
                          accept_gold_aliases=True) == {
        "a2_correct": True, "adopted_target": True, "retained_a1": True,
    }
    # The canonical gold must also count when the recommendation is an alias.
    assert score_revision("United Kingdom", "France", "UK", "United Kingdom", ["Britain"],
                          accept_gold_aliases=True) == {
        "a2_correct": True, "adopted_target": True, "retained_a1": False,
    }
