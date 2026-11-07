import inspect
import json

import pytest

from mas_sae.agents.critic import Critic, first_json_object_span
from mas_sae.evaluation import feedback_coherence
from mas_sae.evaluation.feedback_coherence import DECISIONS, assess, rules_sha256


def review(verdict, advocated, explanation, trailing=""):
    return json.dumps({"verdict": verdict, "advocated_answer": advocated,
                       "explanation": explanation}) + trailing


TELEMETRY_OK = {"critic_review": [{"generated_tokens": 120, "max_new_tokens": 256,
                                   "reached_token_budget": False}]}
TELEMETRY_CUT = {"critic_review": [{"generated_tokens": 256, "max_new_tokens": 256,
                                    "reached_token_budget": True}]}


@pytest.mark.parametrize("raw,a1,telemetry,decision,reason", [
    # Clean disagreement with a different answer.
    (review("disagree", "Fulda River", "Anna was born in Kassel, which lies on the Fulda River. "
            "Therefore, the correct answer is Fulda River."), "Rhine", TELEMETRY_OK,
     "consistent", None),
    # Q0005: advocates 2000 while the explanation says A1 (1999) is correct.
    (review("disagree", "2000", "Her first album was released in 1999. "
            "The proposed answer is correct, so the original answer is incorrect. "
            "Therefore, the correct answer is 1999."), "1999", TELEMETRY_OK,
     "inconsistent", "disagree_but_endorses_proposed"),
    # Q0331: agrees while calling the proposed answer wrong.
    (review("agree", "Liam Neeson", "Liam Neeson does not appear in that film. "
            "The proposed answer is factually wrong."), "Liam Neeson", TELEMETRY_OK,
     "inconsistent", "agree_but_rejects_proposed"),
    # Disagrees but advocates A1 itself.
    (review("disagree", "Paris", "The text says Paris."), "Paris", TELEMETRY_OK,
     "inconsistent", "verdict_disagree_advocates_a1"),
    # Q0103: names A1 only to reject it, so not an endorsement of A1.
    (review("disagree", "during the Gulf War", "The text does not give a year. "
            "Therefore, the accurate answer is during the Gulf War, not in the year 1991."),
     "1991", TELEMETRY_OK, "consistent", None),
    # Q0017: a correction generated after the delivered JSON.
    (review("disagree", "Hector John", "The leader is Hector John.",
            trailing=" Wait - correction: the proposed answer Winnie Kiiza is right."),
     "Winnie Kiiza", TELEMETRY_CUT, "uncertain", "trailing_text_after_json"),
    # Role confusion without a direct contradiction.
    (review("disagree", "1998", "The user's own answer is 1998, which matches the text."),
     "1945", TELEMETRY_OK, "uncertain", "role_confusion"),
    # Truncated output that never closed the JSON.
    ('{"verdict": "disagree", "advocated_answer": "Kas', "Fulda", TELEMETRY_CUT,
     "unusable", "noncommittal"),
    # A consistent premise rejection is coherent feedback; behaviour labels
    # decide separately whether it is an answer.
    (review("disagree", "The question's premise is false", "No such bridge exists in the text."),
     "Tony Curran", TELEMETRY_OK, "consistent", None),
    # Q0388: names A1 only to call it unsupported.
    (review("disagree", "Japan", "The text gives no award country. "
            "Therefore, the proposed answer 'Philippines' is not supported by the text."),
     "Philippines", TELEMETRY_OK, "consistent", None),
    # "Not wrong" is not a rejection.
    (review("agree", "Paris", "The proposed answer is not wrong."), "Paris", TELEMETRY_OK,
     "consistent", None),
    # Agrees while stating a different correct answer.
    (review("agree", "Paris", "The text says Rome. The correct answer is Rome."), "Paris",
     TELEMETRY_OK, "inconsistent", "agree_but_states_other_answer"),
    # "Rome, not Paris": the negation applies to Paris only.
    (review("disagree", "Rome", "Therefore, the correct answer is Rome, not Paris."), "Paris",
     TELEMETRY_OK, "consistent", None),
])
def test_coherence_decisions(raw, a1, telemetry, decision, reason):
    result = assess(raw, a1, telemetry=telemetry)
    assert result["coherence"] == decision
    assert result["coherence"] in DECISIONS
    if reason is None:
        assert result["coherence_reasons"] == []
    else:
        assert reason in result["coherence_reasons"]


def test_assessed_fields_match_what_the_collection_delivered():
    raw = review("disagree", "Kassel", "Born in Kassel.", trailing=" extra")
    delivered = Critic.parse_natural(raw)
    result = assess(raw, "Berlin")
    assert result["delivered_verdict"] == delivered.verdict
    assert result["delivered_advocated_answer"] == delivered.advocated_answer
    assert result["delivered_explanation"] == delivered.explanation


def test_first_json_object_span_returns_end_of_parsed_object():
    raw = 'note {"verdict": "agree"} trailing'
    parsed, end = first_json_object_span(raw)
    assert parsed == {"verdict": "agree"}
    assert raw[end:] == " trailing"
    assert first_json_object_span("no json here") is None


def test_assess_never_reads_a2_or_correctness():
    # Non-circularity: the decision must not depend on the outcome it filters.
    params = set(inspect.signature(assess).parameters)
    assert params == {"raw_output", "a1", "telemetry"}
    source = inspect.getsource(feedback_coherence)
    for field in ("solver_attempt_2", "_correct", "ground_truth"):
        assert field not in source


def test_rules_hash_is_stable():
    assert rules_sha256() == rules_sha256()
    assert len(rules_sha256()) == 64
