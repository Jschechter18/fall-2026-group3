"""Prepare an offline stratified sample, then replay feedback ablations on a GPU."""
import argparse
import json
from pathlib import Path

import yaml

from mas_sae.experiments.adoption_experiment import digest, run_experiment, sample_rows
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.records import read_jsonl, write_jsonl


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
    run = commands.add_parser("run", help="Replay the frozen sample using the configured solver")
    run.add_argument("--sample", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
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
        selected, sampling = sample_rows(read_jsonl(args.interactions), args.partitions, args.size, args.seed)
        packet = {"protocol": "adoption_feedback_ablation_v1", "seed": args.seed,
                  "solver": spec, "dataset": config["dataset"], "sampling": sampling,
                  "input_hashes": {str(p): sha256_file(p) for p in (args.interactions, args.partitions, args.config)},
                  "rows": selected}
        packet["sha256"] = digest(packet)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "sample.json").write_text(json.dumps(packet, indent=2) + "\n")
        write_jsonl(args.output / "sampled_episodes.jsonl", selected)
        print(f"Prepared {len(selected)} questions across {len(sampling)} strata: {args.output / 'sample.json'}")
        return

    packet = json.loads(args.sample.read_text())
    expected = packet.pop("sha256")
    if digest(packet) != expected:
        raise ValueError("Sample/config checksum mismatch")
    if packet["protocol"] != "adoption_feedback_ablation_v1":
        raise ValueError("Unsupported adoption experiment protocol")
    if args.output.exists():
        raise FileExistsError(args.output)
    from mas_sae.agents.solver import Solver
    from mas_sae.data.musique import load_musique_examples_by_id
    from mas_sae.experiments.reproducibility import seed_everything
    from mas_sae.models.roles import configure_agent, load_role_models
    seed_everything(packet["seed"])
    examples = load_musique_examples_by_id("train", [r["question_id"] for r in packet["rows"]],
                                         revision=packet["dataset"]["revision"])
    weights = load_role_models({"solver": packet["solver"]})
    solver = Solver(*weights["solver"])
    configure_agent(solver, packet["solver"])
    provenance = {
        "sample_sha256": expected, "solver": packet["solver"], "dataset": packet["dataset"],
        "revise_template": Solver.REVISE_PROMPT_V1,
        "implementation_sha256": sha256_file(Path(__file__).resolve().parents[2] / "src/mas_sae/experiments/adoption_experiment.py"),
        "cli_sha256": sha256_file(Path(__file__)),
    }
    summary = run_experiment(packet["rows"], {e["id"]: e for e in examples}, solver, args.output, provenance=provenance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
