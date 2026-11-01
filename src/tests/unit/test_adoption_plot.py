from mas_sae.experiments.adoption_plot import summarize_behavior


def test_behavior_plot_counts_source_matches_without_gold_alias_substitution():
    records = [
        dict(question_id='1', solver_attempt_1='guitar', critic_advocated_answer='violin',
             solver_attempt_2='guitar', answer_aliases=['guitar', 'violin'], adopted_target=True),
        dict(question_id='2', solver_attempt_1='Paris', critic_advocated_answer='Rome', solver_attempt_2='Rome'),
        dict(question_id='3', solver_attempt_1='The Paris', critic_advocated_answer='Paris', solver_attempt_2='Paris'),
        dict(question_id='4', solver_attempt_1='Paris', critic_advocated_answer='Rome', solver_attempt_2='Unknown'),
    ]
    row = summarize_behavior('Example', records)
    assert row == dict(method='Example', trials=4, questions=4, adopted=1, retained_a1=1, other=1, overlap=1)
