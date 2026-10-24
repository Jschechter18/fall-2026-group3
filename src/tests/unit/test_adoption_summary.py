import json
from pathlib import Path
import runpy

import pytest

from mas_sae.experiments.records import read_jsonl


ROOT = Path(__file__).resolve().parents[3]
PUBLISHED_RESULTS = ROOT / 'results/adoption_experiment'
PUBLISHED_EPISODES = [
    PUBLISHED_RESULTS / folder / 'episodes.jsonl'
    for folder in ('controlled_100q', 'verification_100q', 'anonymization_100q',
                   'randomized_order_100q')
]


def test_anonymous_randomized_row_reuses_exact_labeled_question_orders(adoption_results_root):
    report = runpy.run_path(str(ROOT / 'scripts/experiments/summarize_adoption_results.py'))
    results = adoption_results_root
    methods, hashes, _ = report['read_methods'](results)
    assert len(methods) == 8
    assignments = json.loads((results / 'randomized_order_100q/order_assignments.json').read_text())
    rows = methods[-1][1]
    assert methods[-1][0] == 'Anonymous candidates: randomized order'
    assert len(rows) == 2 and len({r['question_id'] for r in rows}) == 2
    assert all(r['instruction_condition'] == 'anonymous' and r['candidate_order'] == assignments[r['question_id']] for r in rows)
    labeled = {(r['question_id'], r['arm']): r for r in methods[-2][1]}
    for row in rows:
        other = labeled[(row['question_id'], row['arm'])]
        for key in ('candidate_order', 'question', 'solver_attempt_1', 'critic_advocated_answer', 'context'):
            assert row[key] == other[key]
    assert str(results / 'randomized_order_100q/order_assignments.json') in hashes
    expected_counts = [(1, 1, 0), (2, 0, 0), (0, 2, 0), (2, 0, 0),
                       (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 0)]
    for (name, records), expected in zip(methods, expected_counts):
        summary = report['summarize'](name, records)
        assert (summary['adopted'], summary['retained_a1'], summary['other']) == expected


@pytest.mark.skipif(not all(path.exists() for path in PUBLISHED_EPISODES),
                    reason='published per-row results are stored in S3')
def test_published_rows_retain_their_alias_aware_scores():
    report = runpy.run_path(str(ROOT / 'scripts/experiments/summarize_adoption_results.py'))
    methods, _, _ = report['read_methods'](PUBLISHED_RESULTS)
    expected_counts = [(89, 4, 5), (86, 8, 4), (35, 61, 2), (36, 59, 3),
                       (10, 73, 15), (26, 50, 22), (32, 65, 1)]
    for (name, records), expected in zip(methods[:7], expected_counts):
        summary = report['summarize'](name, records)
        assert (summary['adopted'], summary['retained_a1'], summary['other']) == expected
