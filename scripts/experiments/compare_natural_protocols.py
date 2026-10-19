"""Prepare/run the exact plotted cohort with saved and repaired natural feedback."""
import argparse
import json
from pathlib import Path

import yaml

from mas_sae.experiments.adoption_experiment import digest, load_sample
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.natural_protocol_comparison import (
    needs_feedback, prepare_feedback, run_conditions, select_cohort, summarize_method,
)
from mas_sae.experiments.records import append_jsonl, read_jsonl, write_csv
from summarize_adoption_results import read_methods, render_chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('results/adoption_experiment'))
    parser.add_argument('--output', type=Path, default=Path('results/adoption_experiment/adoption_results'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true', help='Redraw completed results without loading models')
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'),
                        help='Default: original precision on CUDA; float32 on MPS/CPU')
    parser.add_argument('--critic-max-new-tokens', type=int, default=1024,
                        help='Larger budget for regenerating previously unusable feedback')
    args = parser.parse_args()
    root, output = args.results_root, args.output
    if args.plot_only:
        summary = json.loads((output / 'summary.json').read_text())['methods']
        render_chart(output, summary, natural=True)
        print(f"Saved {output / 'adoption_comparison.svg'} and .png")
        return
    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if not args.prepare_only:
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; choose --device mps or --device cpu.')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable in this PyTorch installation or on this host.')
    packet, sample_hash = load_sample(root / 'sample_100q/sample.json')
    # Reuse all pairing/provenance checks from the fixed plot before selecting its cohort.
    methods, input_hashes, checked_hash = read_methods(root)
    if checked_hash != sample_hash:
        raise ValueError('Fixed plot and frozen sample differ')
    assignments = json.loads((root / 'randomized_order_100q/order_assignments.json').read_text())
    controlled = read_jsonl(root / 'controlled_100q/episodes.jsonl')
    rows = select_cohort(packet, controlled, assignments)
    if {r['question_id'] for r in rows} != {r['question_id'] for r in methods[0][1]}:
        raise ValueError('Cohort differs from the fixed plot')
    configs = [Path(p) for p in packet['input_hashes'] if p.endswith('.yaml')]
    if len(configs) != 1 or sha256_file(configs[0]) != packet['input_hashes'][str(configs[0])]:
        raise ValueError('Original collection config is missing or changed')
    from mas_sae.models.roles import resolve_roles
    roles = resolve_roles(yaml.safe_load(configs[0].read_text()))
    critic_spec = {**roles['critic'], 'generation': {**roles['critic']['generation'],
                                                   'max_new_tokens': args.critic_max_new_tokens}}
    solver_spec = dict(packet['solver'])
    for spec in (solver_spec, critic_spec):
        spec.pop('device_map', None)
        spec['device'] = 'cuda:0' if device == 'cuda' else device
        spec['dtype'] = args.dtype or (spec['dtype'] if device == 'cuda' else 'float32')
    manifest = dict(protocol='natural_eight_protocol_comparison_v1', original_sample_sha256=sample_hash,
                    input_hashes=input_hashes, solver=solver_spec, critic=critic_spec, dataset=packet['dataset'],
                    original_solver=packet['solver'],
                    question_ids=[r['question_id'] for r in rows],
                    regenerate_question_ids=[r['question_id'] for r in rows if needs_feedback(r)],
                    order_assignments={r['question_id']: assignments[r['question_id']] for r in rows},
                    feedback_policy='Reuse usable saved natural feedback; regenerate missing feedback only. No gold targets.',
                    randomized_policy='Select fixed-order responses using the original question-level assignments.',
                    scoring='Alias-aware adoption as in fixed plot; matches-both separated in stacked bars.',
                    implementation_sha256=sha256_file(Path(__file__).resolve().parents[2] / 'src/mas_sae/experiments/natural_protocol_comparison.py'),
                    cli_sha256=sha256_file(Path(__file__)))
    manifest['sha256'] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'plan.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        previous = json.loads(manifest_path.read_text())
        code_fields = {'sha256', 'implementation_sha256', 'cli_sha256'}
        same_settings = ({k: v for k, v in previous.items() if k not in code_fields} ==
                         {k: v for k, v in manifest.items() if k not in code_fields})
        feedback_only_resume = (same_settings and not (output / 'episodes.jsonl').exists()
                                and not (output / 'summary.json').exists())
        if not feedback_only_resume and any((output / name).exists() for name in ('feedback.jsonl', 'episodes.jsonl', 'summary.json')):
            raise ValueError('Existing inference plan differs; use a separate output directory')
        if feedback_only_resume:
            append_jsonl(output / 'plan_history.jsonl', previous)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Prepared {len(rows)} exact questions; {len(manifest['regenerate_question_ids'])} need natural feedback.", flush=True)
    if args.prepare_only:
        return
    from mas_sae.agents.critic import Critic
    from mas_sae.agents.solver import Solver
    from mas_sae.data.musique import load_musique_examples_by_id
    from mas_sae.experiments.anonymized_adoption import CandidateSolver
    from mas_sae.experiments.reproducibility import seed_everything
    from mas_sae.experiments.verification_adoption import EvidenceVerificationSolver
    from mas_sae.models.roles import configure_agent, load_role_models
    seed_everything(packet['seed'])
    examples = {e['id']: e for e in load_musique_examples_by_id(
        'train', manifest['question_ids'], revision=packet['dataset']['revision'])}
    sources = {r['question_id']: r for r in controlled if r['status'] == 'completed'}
    for row in rows:
        ex, source = examples[row['question_id']], sources[row['question_id']]
        if (row['question'] != ex['question'] or row['ground_truth'] != ex['answer'] or
                row['solver_attempt_1'] != source['solver_attempt_1'] or
                Solver.format_paragraphs(ex['paragraphs']) != source['context']):
            raise ValueError('Pinned question, A1 or passages differ from the fixed plot')
    # Load roles sequentially so both models need not fit on the GPU together.
    weights = load_role_models({'critic': critic_spec})
    critic = Critic(*weights['critic'])
    configure_agent(critic, critic_spec)
    feedback = prepare_feedback(rows, examples, critic, output / 'feedback.jsonl')
    del critic, weights
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
    weights = load_role_models({'solver': solver_spec})
    agents = {name: cls(*weights['solver']) for name, cls in
              (('original', Solver), ('verification', EvidenceVerificationSolver), ('candidate', CandidateSolver))}
    for agent in agents.values():
        configure_agent(agent, solver_spec)
    methods = run_conditions(rows, examples, feedback, assignments, agents, output / 'episodes.jsonl')
    summary = [summarize_method(name, records) for name, records in methods]
    write_csv(output / 'method_summary.csv', summary)
    render_chart(output, summary, natural=True)
    (output / 'summary.json').write_text(json.dumps({'methods': summary, 'plan_sha256': manifest['sha256']}, indent=2) + '\n')
    print(f"Saved {output / 'adoption_comparison.svg'}")


if __name__ == '__main__':
    main()
