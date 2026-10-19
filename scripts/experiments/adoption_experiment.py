"""Prepare a frozen sample and run feedback, instruction, and anonymity experiments."""
import argparse
import json
import logging
from pathlib import Path

import yaml

from mas_sae.experiments.adoption_experiment import digest, load_sample, run_experiment, sample_rows
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.records import read_jsonl, write_jsonl


def experiment_source_sha(filename):
    """Hash an experiment implementation relative to this CLI's repository."""
    directory = Path(__file__).resolve().parents[2] / "src/mas_sae/experiments"
    return sha256_file(directory / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Offline sampling; no model or dataset download")
    prepare.add_argument("--interactions", type=Path, required=True)
    prepare.add_argument("--partitions", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True, help="Original collection YAML: solver and dataset settings")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--size", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)
    natural_prepare = commands.add_parser("prepare-natural", help="Freeze a Natural train sample without selecting on A2; optionally apply #115 coherence")
    for name in ("interactions", "partitions", "config", "output"):
        natural_prepare.add_argument("--" + name, type=Path, required=True)
    natural_prepare.add_argument("--audit", type=Path, help="Optional #115 train audit CSV for additional coherence screening")
    natural_prepare.add_argument("--size", type=int, default=50)
    natural_prepare.add_argument("--seed", type=int, default=42)
    natural_prepare.add_argument("--exclude-sample", type=Path, help="Prior sample.json whose questions must be excluded")
    matched_prepare = commands.add_parser("prepare-natural-matched", help="Reuse original questions and order; no audit or resampling")
    for name in ("sample", "controlled-run", "randomized-run", "output"):
        matched_prepare.add_argument("--" + name, type=Path, required=True)
    natural_run = commands.add_parser("run-natural", help="Paired original and randomized labeled A2 using saved Natural recommendations")
    natural_run.add_argument("--sample", type=Path, required=True)
    natural_run.add_argument("--output", type=Path, required=True)
    natural_run.add_argument("--order-seed", type=int, default=42)
    run = commands.add_parser("run", help="Replay the frozen sample using the configured solver")
    run.add_argument("--sample", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    controlled = commands.add_parser("run-controlled", help="Compare wrong recommendations on correct A1s and gold corrections on wrong A1s")
    controlled.add_argument("--sample", type=Path, required=True, help="Existing prepared sample.json")
    controlled.add_argument("--output", type=Path, required=True, help="New directory, separate from ablation results")
    verification = commands.add_parser("run-verification", help="Replay saved controlled feedback under original and evidence-verification instructions")
    verification.add_argument("--sample", type=Path, required=True)
    verification.add_argument("--controlled-run", type=Path, required=True, help="Completed run-controlled output directory")
    verification.add_argument("--output", type=Path, required=True, help="New output directory including results.csv and comparisons.csv")
    anonymization = commands.add_parser("run-anonymization", help="Compare anonymous and source-labeled candidate answers in both orders")
    anonymization.add_argument("--sample", type=Path, required=True)
    anonymization.add_argument("--controlled-run", type=Path, required=True)
    anonymization.add_argument("--output", type=Path, required=True)
    randomized = commands.add_parser("run-randomized-order", help="Labeled candidate evaluation with a frozen balanced random order per question")
    randomized.add_argument("--sample", type=Path, required=True)
    randomized.add_argument("--controlled-run", type=Path, required=True)
    randomized.add_argument("--order-seed", type=int, default=42)
    randomized.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare-natural-matched":
        from mas_sae.experiments.matched_adoption import prepare_matched_sample
        packet = prepare_matched_sample(args.sample, args.controlled_run, args.randomized_run, args.output)
        print(json.dumps(packet["screening"], indent=2))
        print(f"Prepared matched sample: {args.output / 'sample.json'}")
        return
    if args.command in {"prepare", "prepare-natural"}:
        config = yaml.safe_load(args.config.read_text())
        # Validate before producing any artifacts; only the solver is loaded later.
        from mas_sae.models.roles import resolve_role_spec
        from mas_sae.experiments.reproducibility import validate_commit_revision
        spec = resolve_role_spec("solver", config["roles"]["solver"], default_max_new_tokens=32)
        validate_commit_revision(spec["revision"], "solver")
        validate_commit_revision(config["dataset"]["revision"], "dataset")
        if spec["generation"]["do_sample"]:
            raise ValueError("This paired diagnostic requires deterministic decoding")
        if config["dataset"]["source_split"] != "train":
            raise ValueError("Use the original MuSiQue-train collection config")
        source_rows = read_jsonl(args.interactions)
        natural_metadata = {}
        if args.command == "prepare-natural":
            from mas_sae.experiments.natural_adoption import SAMPLE_PROTOCOL, prepare_natural_pool
            source_rows, natural_metadata = prepare_natural_pool(
                source_rows, args.partitions, args.audit, args.exclude_sample,
            )
        selected, sampling = sample_rows(source_rows, args.partitions, args.size, args.seed)
        packet = {"protocol": "adoption_feedback_ablation_v1", "seed": args.seed,
                  "solver": spec, "dataset": config["dataset"], "sampling": sampling,
                  "input_hashes": {str(p): sha256_file(p) for p in (args.interactions, args.partitions, args.config)},
                  "rows": selected}
        if args.command == "prepare-natural":
            packet.update(protocol=SAMPLE_PROTOCOL, **natural_metadata)
            if args.audit is not None:
                packet["input_hashes"][str(args.audit)] = sha256_file(args.audit)
            if args.exclude_sample:
                packet["input_hashes"][str(args.exclude_sample)] = sha256_file(args.exclude_sample)
        packet["sha256"] = digest(packet)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "sample.json").write_text(json.dumps(packet, indent=2) + "\n")
        write_jsonl(args.output / "sampled_episodes.jsonl", selected)
        print(f"Prepared {len(selected)} questions across {len(sampling)} strata: {args.output / 'sample.json'}")
        return

    packet, expected = load_sample(args.sample)
    wanted_protocol = "adoption_natural_sample_v1" if args.command == "run-natural" else "adoption_feedback_ablation_v1"
    if packet["protocol"] != wanted_protocol:
        raise ValueError("Unsupported adoption experiment protocol")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "run-natural" and packet.get("cohort_mode") == "matched_original" and args.order_seed != packet["order_seed"]:
        raise ValueError("Matched experiment must retain the original order seed")
    from mas_sae.agents.solver import Solver
    from mas_sae.data.musique import load_musique_examples_by_id
    from mas_sae.experiments.reproducibility import seed_everything
    from mas_sae.models.roles import configure_agent, load_role_models
    if args.command in {"run-verification", "run-anonymization", "run-randomized-order"}:
        from mas_sae.experiments.verification_adoption import load_controlled_source, validate_sources
        source_records, source_hashes = load_controlled_source(args.controlled_run, packet, expected)
    seed_everything(packet["seed"])
    examples = load_musique_examples_by_id("train", [r["question_id"] for r in packet["rows"]],
                                         revision=packet["dataset"]["revision"])
    examples = {e["id"]: e for e in examples}
    if args.command in {"run-verification", "run-anonymization", "run-randomized-order"}:
        validate_sources(packet["rows"], examples, source_records)
    if args.command == "run-natural":
        # Validate pinned input before loading model weights.
        for row in packet["rows"]:
            if row["question"] != examples[row["question_id"]]["question"] or row["ground_truth"] != examples[row["question_id"]]["answer"]:
                raise ValueError("Natural sample differs from pinned dataset")
    weights = load_role_models({"solver": packet["solver"]})
    solver = Solver(*weights["solver"])
    configure_agent(solver, packet["solver"])
    provenance = {
        "sample_sha256": expected, "solver": packet["solver"], "dataset": packet["dataset"],
        "revise_template": Solver.REVISE_PROMPT_V1,
        "implementation_sha256": experiment_source_sha("adoption_experiment.py"),
        "cli_sha256": sha256_file(Path(__file__)),
    }
    if args.command in {"run-natural", "run-anonymization", "run-randomized-order"}:
        from mas_sae.experiments.anonymized_adoption import CandidateSolver
        candidate_solver = CandidateSolver(*weights["solver"])
        configure_agent(candidate_solver, packet["solver"])
    if args.command == "run-natural":
        from mas_sae.experiments.natural_adoption import run_natural
        provenance.update(screening=packet["screening"], input_hashes=packet["input_hashes"],
                          implementation_hashes={name: experiment_source_sha(name)
                              for name in ("natural_adoption.py", "randomized_adoption.py", "anonymized_adoption.py")})
        matched = packet.get("cohort_mode") == "matched_original"
        if matched:
            provenance.update(cohort_mode=packet["cohort_mode"], original_sample_sha256=packet["original_sample_sha256"],
                              matched_question_ids=[r["question_id"] for r in packet["rows"]])
        summary = run_natural(packet["rows"], examples, solver, candidate_solver,
                              args.output, seed=args.order_seed, provenance=provenance,
                              assigned_orders=packet.get("order_assignments"), require_conflict=not matched)
    elif args.command == "run-controlled":
        from mas_sae.experiments.controlled_adoption import (
            ControlledProposer, GENERATOR_RULES, GENERATOR_TOKENS, PROPOSAL_PROMPT, PROTOCOL, run_controlled,
        )
        # Independent, stateless agent with shared weights; Solver settings stay intact.
        proposer_spec = {**packet["solver"], "generation": {
            "do_sample": False, "max_new_tokens": GENERATOR_TOKENS,
        }}
        proposer = ControlledProposer(*weights["solver"])
        configure_agent(proposer, proposer_spec)
        provenance.update(protocol=PROTOCOL, target_generator=proposer_spec,
                          target_generator_prompt=PROPOSAL_PROMPT, target_generator_rules=GENERATOR_RULES,
                          controlled_implementation_sha256=experiment_source_sha("controlled_adoption.py"),
                          controlled_targets_sha256=experiment_source_sha("controlled_targets.py"))
        summary = run_controlled(packet["rows"], examples, solver, proposer,
                                 args.output, provenance=provenance)
    elif args.command == "run-verification":
        from mas_sae.experiments.verification_adoption import EvidenceVerificationSolver, run_verification
        verification_solver = EvidenceVerificationSolver(*weights["solver"])
        configure_agent(verification_solver, packet["solver"])
        provenance.update(controlled_run=str(args.controlled_run), source_hashes=source_hashes,
                          verification_implementation_sha256=experiment_source_sha("verification_adoption.py"))
        summary = run_verification(packet["rows"], examples, source_records,
                                   {"original": solver, "evidence_verification": verification_solver},
                                   args.output, provenance=provenance)
    elif args.command == "run-anonymization":
        from mas_sae.experiments.anonymized_adoption import run_anonymization
        provenance.update(controlled_run=str(args.controlled_run), source_hashes=source_hashes,
                          anonymization_implementation_sha256=experiment_source_sha("anonymized_adoption.py"))
        summary = run_anonymization(packet["rows"], examples, source_records,
                                    candidate_solver, args.output, provenance=provenance)
    elif args.command == "run-randomized-order":
        from mas_sae.experiments.randomized_adoption import run_randomized_order
        provenance.update(controlled_run=str(args.controlled_run), source_hashes=source_hashes,
                          randomized_implementation_sha256=experiment_source_sha("randomized_adoption.py"),
                          candidate_implementation_sha256=experiment_source_sha("anonymized_adoption.py"))
        summary = run_randomized_order(packet["rows"], examples, source_records,
                                       candidate_solver, args.output, seed=args.order_seed, provenance=provenance)
    else:
        summary = run_experiment(packet["rows"], examples, solver, args.output, provenance=provenance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
