from mas_sae.agents.critic import Critic
from mas_sae.experiments.natural_protocol_comparison import (
    METHODS, prepare_feedback, run_conditions, summarize_method,
)


class FakeAgent:
    last_generation = {}

    def build_revise_prompt(self, question, paragraphs, a1, feedback):
        return 'Rome'

    def build_candidate_prompt(self, question, paragraphs, a1, target, condition, order):
        return a1 if order == 'a1_first' else target

    def _generate(self, prompt):
        return prompt


def test_all_eight_rows_share_questions_and_original_orders_and_resume(tmp_path):
    rows = [dict(question_id=str(i), question='Capital?', solver_attempt_1='Paris') for i in range(2)]
    examples = {r['question_id']: dict(answer='Rome', paragraphs=[]) for r in rows}
    feedback = {r['question_id']: dict(critic_advocated_answer='Rome', critic_feedback='Use Rome',
                                     source='saved_natural') for r in rows}
    assignments = {'0': 'a1_first', '1': 'critic_first'}
    agents = {key: FakeAgent() for key in ('original', 'verification', 'candidate')}
    path = tmp_path / 'episodes.jsonl'
    methods = run_conditions(rows, examples, feedback, assignments, agents, path)
    assert [name for name, _ in methods] == list(METHODS)
    assert all({r['question_id'] for r in records} == {'0', '1'} for _, records in methods)
    for _, records in methods[6:]:
        assert [r['solver_attempt_2'] for r in records] == ['Paris', 'Rome']
    assert len(path.read_text().splitlines()) == 12
    assert run_conditions(rows, examples, feedback, assignments, {}, path) == methods


def test_overlap_is_not_double_counted_and_empty_rate_is_unavailable():
    row = dict(question_id='1', solver_attempt_1='Rome', critic_advocated_answer='Rome',
               adopted_target=True, retained_a1=True, a2_correct=True,
               recommendation_correct=True, recommendation_source='saved_natural')
    summary = summarize_method('example', [row])
    assert summary['overlap'] == 1
    assert summary['adopted'] == summary['retained_a1'] == summary['other'] == 0
    assert summary['correct_accept_pct'] == 100
    assert summary['incorrect_accept_pct'] != summary['incorrect_accept_pct']  # NaN, not a false 0%.


def test_summary_separates_disagreements_and_feedback_generation_cohorts():
    rows = [
        dict(question_id='saved', solver_attempt_1='Paris', critic_advocated_answer='Rome',
             adopted_target=True, retained_a1=False, a2_correct=True,
             recommendation_correct=True, recommendation_source='saved_natural'),
        dict(question_id='regenerated', solver_attempt_1='Paris', critic_advocated_answer='Paris',
             adopted_target=True, retained_a1=True, a2_correct=False,
             recommendation_correct=False, recommendation_source='regenerated_natural'),
    ]
    summary = summarize_method('example', rows)
    assert summary['a2_correct'] == 1
    assert summary['correct_disagreement_trials'] == 1
    assert summary['correct_disagreement_adopted'] == 1
    assert summary['saved_natural_trials'] == 1
    assert summary['saved_natural_correct_disagreement_accept_pct'] == 100
    assert summary['regenerated_natural_trials'] == 1
    assert summary['regenerated_natural_wrong_accept_pct'] == 100


def test_only_missing_feedback_is_regenerated_and_checkpointed(tmp_path):
    raw = '{"verdict":"disagree","advocated_answer":"Rome","explanation":"Evidence"}'
    rows = [dict(question_id='saved', critic_raw_output=raw),
            dict(question_id='missing', question='Capital?', solver_attempt_1='Paris', critic_raw_output='bad JSON')]

    class FakeCritic:
        calls = 0

        def critique(self, **kwargs):
            self.calls += 1
            assert 'advocated_answer' not in kwargs
            return Critic.parse_natural(raw)

    critic = FakeCritic()
    path = tmp_path / 'feedback.jsonl'
    feedback = prepare_feedback(rows, {'missing': {'paragraphs': []}}, critic, path)
    assert critic.calls == 1
    assert feedback['saved']['source'] == 'saved_natural'
    assert feedback['missing']['source'] == 'regenerated_natural'
    assert prepare_feedback(rows, {}, None, path) == feedback


def test_truncated_json_retries_with_compact_prompt_and_preserves_failure(tmp_path):
    class TruncatedCritic:
        last_generation = {'reached_token_budget': True}

        def critique(self, **kwargs):
            return Critic.parse_natural('{"verdict":"disagree","advocated_answer":"180","explanation":"unfinished')

        def build_critique_prompt(self, **kwargs):
            assert 'advocated_answer' not in kwargs
            return 'Original natural prompt'

        def _generate(self, prompt):
            assert 'one short sentence' in prompt
            self.last_generation = {'reached_token_budget': False}
            return '{"verdict":"disagree","advocated_answer":"Rome","explanation":"Passage evidence."}'

    rows = [dict(question_id='q', question='Capital?', solver_attempt_1='Paris', critic_raw_output='invalid')]
    path = tmp_path / 'feedback.jsonl'
    feedback = prepare_feedback(rows, {'q': {'paragraphs': []}}, TruncatedCritic(), path)
    assert feedback['q']['critic_advocated_answer'] == 'Rome'
    assert feedback['q']['retry_attempt'] == 1
    assert '180' in (tmp_path / 'failed_feedback.jsonl').read_text()
    assert prepare_feedback(rows, {}, None, path) == feedback


def test_failed_retries_remain_bounded_without_checkpointing_invalid_feedback(tmp_path):
    class InvalidCritic:
        calls = 0

        def critique(self, **kwargs):
            return Critic.parse_natural('invalid')

        def build_critique_prompt(self, **kwargs):
            return 'Natural prompt'

        def _generate(self, prompt):
            self.calls += 1
            return 'invalid'

    critic = InvalidCritic()
    path = tmp_path / 'feedback.jsonl'
    rows = [dict(question_id='q', question='Capital?', solver_attempt_1='Paris', critic_raw_output='invalid')]
    try:
        prepare_feedback(rows, {'q': {'paragraphs': []}}, critic, path)
    except ValueError as error:
        assert '3 attempts' in str(error)
    else:
        raise AssertionError('Invalid feedback must stop the run')
    assert critic.calls == 2
    assert not path.exists()
    assert len((tmp_path / 'failed_feedback.jsonl').read_text().splitlines()) == 3
