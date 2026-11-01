from mas_sae.evaluation.scoring import interaction_labels


def test_accepts_correct_advocated_answer():
    labels = interaction_labels(
        attempt1="London",
        attempt2="Paris",
        advocated_answer="Paris",
        gold="Paris",
    )

    assert labels["solver_accepted_feedback"] is True
    assert labels["solver_changed_answer"] is True
    assert labels["critic_feedback_correct"] is True
    assert labels["solver_attempt_2_correct"] is True


def test_rejects_advocated_answer():
    labels = interaction_labels(
        attempt1="London",
        attempt2="London",
        advocated_answer="Paris",
        gold="Paris",
    )

    assert labels["solver_accepted_feedback"] is False
    assert labels["solver_changed_answer"] is False
    assert labels["critic_feedback_correct"] is True
    assert labels["solver_attempt_2_correct"] is False


def test_noncommittal_feedback_has_no_acceptance_label():
    labels = interaction_labels(
        attempt1="London",
        attempt2="London",
        advocated_answer=None,
        gold="Paris",
    )

    assert labels["solver_accepted_feedback"] is None
    assert labels["critic_feedback_correct"] is None


def test_alias_is_accepted():
    labels = interaction_labels(
        attempt1="Something else",
        attempt2="NYC",
        advocated_answer="New York City",
        gold="New York",
        aliases=["New York City", "NYC"],
    )

    assert labels["solver_accepted_feedback"] is True
    assert labels["critic_feedback_correct"] is True
    assert labels["solver_attempt_2_correct"] is True
