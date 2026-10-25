"""Reuse the original question sample for Natural protocol comparisons."""
import json
from pathlib import Path

from mas_sae.agents.critic import Critic
from mas_sae.experiments.adoption_experiment import digest, load_sample
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.natural_adoption import SAMPLE_PROTOCOL
from mas_sae.experiments.randomized_adoption import ORDERS, PROTOCOL as ORDER_PROTOCOL
from mas_sae.experiments.records import write_jsonl
from mas_sae.experiments.verification_adoption import load_controlled_source


def prepare_matched_sample(sample_path, controlled_directory, randomized_directory, output_dir):
    """Freeze the shared available questions; no resampling or audit screening.

    Natural agreement is kept. Missing Natural recommendations and unavailable
    controlled targets are recorded before any new A2 is generated. Order comes
    from the original randomized controlled run rather than being reassigned.
    """
    packet, original_hash = load_sample(sample_path)
    if packet['protocol'] != 'adoption_feedback_ablation_v1':
        raise ValueError('Expected original adoption experiment sample')
    sources, controlled_hashes = load_controlled_source(controlled_directory, packet, original_hash)
    randomized_directory = Path(randomized_directory)
    order_metadata = json.loads((randomized_directory / 'provenance.json').read_text())
    if order_metadata['protocol'] != ORDER_PROTOCOL or order_metadata['sample_sha256'] != original_hash:
        raise ValueError('Order assignments must come from the same original sample')
    assignments = json.loads((randomized_directory / 'order_assignments.json').read_text())
    if set(assignments) != {r['question_id'] for r in packet['rows']} or any(v not in ORDERS for v in assignments.values()):
        raise ValueError('Original order assignments must cover every original question')
    completed_ids = {r['question_id'] for r in sources if r['status'] == 'completed'}
    selected, exclusions = [], []
    for row in packet['rows']:
        qid = row['question_id']
        if qid not in completed_ids:
            exclusions.append(dict(question_id=qid, reason='no_completed_controlled_trial'))
            continue
        feedback = Critic.parse_natural(row['critic_raw_output'])
        if feedback.to_solver_text() != row['critic_feedback'] or feedback.advocated_answer != row.get('critic_advocated_answer'):
            raise ValueError(f'Saved Natural delivery differs from parsed source: {qid}')
        if feedback.noncommittal or not feedback.advocated_answer:
            exclusions.append(dict(question_id=qid, reason='missing_natural_recommendation'))
            continue
        selected.append({**row, 'natural_screen': {
            'coherence': 'not_assessed', 'coherence_reasons': '',
            'eligibility': 'shared_source_availability_only_no_audit',
        }})
    if not selected:
        raise ValueError('No shared available questions')
    input_hashes = dict(packet['input_hashes'])
    input_hashes[str(sample_path)] = sha256_file(sample_path)
    for name in ('provenance.json', 'order_assignments.json', 'summary.json'):
        path = randomized_directory / name
        input_hashes[str(path)] = sha256_file(path)
    input_hashes.update({str(Path(controlled_directory) / name): value for name, value in controlled_hashes.items()})
    result = dict(protocol=SAMPLE_PROTOCOL, seed=packet['seed'], solver=packet['solver'], dataset=packet['dataset'],
                  rows=selected, input_hashes=input_hashes, cohort_mode='matched_original',
                  original_sample_sha256=original_hash,
                  order_seed=order_metadata['order_seed'],
                  order_assignments={r['question_id']: assignments[r['question_id']] for r in selected},
                  screening={'method': 'shared_source_availability_no_audit', 'original_questions': len(packet['rows']),
                             'controlled_completed_questions': len(completed_ids), 'shared_questions': len(selected),
                             'exclusions': exclusions})
    result['sha256'] = digest(result)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sample.json').write_text(json.dumps(result, indent=2) + '\n')
    write_jsonl(output / 'sampled_episodes.jsonl', selected)
    write_jsonl(output / 'excluded_questions.jsonl', exclusions)
    return result


def align_matched_methods(methods, natural_records, provenance, original_sample_sha256):
    """Restrict every controlled method to the same predeclared Natural questions.

    Availability is frozen in preparation, not selected from observed A2 outcomes.
    Recommendations differ by source, while question, A1 and passages must match.
    """
    if provenance.get('cohort_mode') != 'matched_original' or provenance.get('original_sample_sha256') != original_sample_sha256:
        raise ValueError('Natural run must reuse the original controlled sample, without audit screening')
    if provenance.get('screening', {}).get('method') != 'shared_source_availability_no_audit':
        raise ValueError('Matched comparison must not use audit-screened or resampled questions')
    expected = set(provenance['matched_question_ids'])
    if not expected or {r['question_id'] for r in natural_records} != expected:
        raise ValueError('Natural run does not cover its declared matched cohort')
    natural = {r['question_id']: r for r in natural_records}
    result = []
    for name, rows in methods:
        selected = [r for r in rows if r['question_id'] in expected]
        if {r['question_id'] for r in selected} != expected:
            raise ValueError(f'Controlled method does not cover shared questions: {name}')
        for row in selected:
            for key in ('question', 'solver_attempt_1', 'context', 'ground_truth'):
                if row[key] != natural[row['question_id']][key]:
                    raise ValueError(f'Matched input differs: {name}, {row["question_id"]}, {key}')
        result.append((name, selected))
    return result
