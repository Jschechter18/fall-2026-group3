import json
from pathlib import Path

import pytest

from mas_sae.agents.solver import Solver
from mas_sae.experiments.adoption_experiment import load_sample
from mas_sae.experiments.anonymized_adoption import CandidateSolver
from mas_sae.experiments.matched_adoption import prepare_matched_sample, align_matched_methods
from mas_sae.experiments.natural_adoption import run_natural
from mas_sae.experiments.records import read_jsonl

def test_matched_preparation_reuses_original_questions_and_orders_without_screening(
        tmp_path, adoption_results_root):
    result = prepare_matched_sample(adoption_results_root/'sample_100q/sample.json',
                                    adoption_results_root/'controlled_100q',
                                    adoption_results_root/'randomized_order_100q',
                                    tmp_path/'matched')
    original, checksum = load_sample(adoption_results_root/'sample_100q/sample.json')
    sources = read_jsonl(adoption_results_root/'controlled_100q/episodes.jsonl')
    controlled_ids = {r['question_id'] for r in sources if r['status']=='completed'}
    expected = [r for r in original['rows'] if r['question_id'] in controlled_ids
                and not r['critic_noncommittal'] and r.get('critic_advocated_answer')]
    assert [r['question_id'] for r in result['rows']] == [r['question_id'] for r in expected]
    assert result['original_sample_sha256'] == checksum
    assert result['screening']['method'] == 'shared_source_availability_no_audit'
    assert len(result['rows']) + len(result['screening']['exclusions']) == len(original['rows'])
    assert all(r['natural_screen']['coherence'] == 'not_assessed' for r in result['rows'])
    assert any(r['solver_attempt_1'] == r['critic_advocated_answer'] for r in result['rows'])
    old_orders=json.loads((adoption_results_root/'randomized_order_100q/order_assignments.json').read_text())
    assert result['order_assignments']=={r['question_id']:old_orders[r['question_id']] for r in expected}


def test_alignment_rejects_unrelated_sample_and_matches_inputs():
    base=dict(question_id='a', question='Where?', solver_attempt_1='Paris', context='Source', ground_truth='Rome')
    methods=[('Method', [dict(base, arm='one'), dict(base, arm='two'), dict(base, question_id='other')])]
    metadata=dict(cohort_mode='matched_original', original_sample_sha256='same', matched_question_ids=['a'],
                  screening={'method':'shared_source_availability_no_audit'})
    aligned=align_matched_methods(methods,[base],metadata,'same')
    assert len(aligned[0][1])==2
    with pytest.raises(ValueError,match='original controlled sample'):
        align_matched_methods(methods,[base],metadata,'different')
    with pytest.raises(ValueError,match='Matched input differs'):
        align_matched_methods(methods,[dict(base, solver_attempt_1='Berlin')],metadata,'same')


def test_matched_runner_keeps_agreement_and_frozen_order(tmp_path):
    row=dict(question_id='2hop__one', question='Where?', ground_truth='Paris', solver_attempt_1='Paris',
             critic_advocated_answer='Paris', critic_condition='natural', critic_feedback='Paris',
             natural_screen={'coherence':'not_assessed'})
    example=dict(question='Where?', answer='Paris', answer_aliases=[], paragraphs=[])
    class Original(Solver):
        def __init__(self): self.last_generation={}
        def _generate(self, prompt): return 'Paris'
    class Candidate(CandidateSolver):
        def __init__(self): self.last_generation={}
        def _generate(self, prompt):
            assert "Candidate A (the reviewer's recommendation)" in prompt
            return 'Paris'
    summary=run_natural([row],{row['question_id']:example},Original(),Candidate(),tmp_path/'run',
                        assigned_orders={row['question_id']:'critic_first'},require_conflict=False)
    assert summary['conditions']['original']['ambiguous_overlap']==1
    assert summary['conditions']['labeled_randomized']['ambiguous_overlap']==1
