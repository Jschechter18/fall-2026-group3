import json
from pathlib import Path
import runpy

from mas_sae.experiments.records import read_jsonl


def test_anonymous_randomized_row_reuses_exact_labeled_question_orders():
    root = Path(__file__).resolve().parents[3]
    report = runpy.run_path(str(root / 'scripts/experiments/summarize_adoption_results.py'))
    results = root / 'results/adoption_experiment'
    methods, hashes, _ = report['read_methods'](results)
    assert len(methods) == 8
    assignments = json.loads((results / 'randomized_order_100q/order_assignments.json').read_text())
    rows = methods[-1][1]
    assert methods[-1][0] == 'Anonymous candidates: randomized order'
    assert len(rows) == 98 and len({r['question_id'] for r in rows}) == 80
    assert all(r['instruction_condition'] == 'anonymous' and r['candidate_order'] == assignments[r['question_id']] for r in rows)
    labeled = {(r['question_id'], r['arm']): r for r in methods[-2][1]}
    for row in rows:
        other = labeled[(row['question_id'], row['arm'])]
        for key in ('candidate_order', 'question', 'solver_attempt_1', 'critic_advocated_answer', 'context'):
            assert row[key] == other[key]
    assert str(results / 'randomized_order_100q/order_assignments.json') in hashes
    # Existing rows retain their original alias-aware scores.
    old_counts = [(89, 4, 5), (86, 8, 4), (35, 61, 2), (36, 59, 3),
                  (10, 73, 15), (26, 50, 22), (32, 65, 1)]
    for (name, records), expected in zip(methods[:7], old_counts):
        summary = report['summarize'](name, records)
        assert (summary['adopted'], summary['retained_a1'], summary['other']) == expected
