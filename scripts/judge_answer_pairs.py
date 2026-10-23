"""Ask the run's validator model whether two answers to a question name the same thing.

Under a label policy with candidate rules, a label needs a judge verdict for
every non-exact relation between answer-shaped texts (see
``mas_sae.evaluation.behavior_labels``). This reads a run's interactions, asks
the judge only about the pairs each episode's label lists in ``judgment_pairs``,
and writes into a new folder:

    judgments.jsonl   one record per (episode_id, pair): both texts, the verdict
                      (same / different / uncertain, or null when the reply was
                      none of these), the raw reply and generation telemetry
    provenance.json   judge model, prompt hash, label policy, input hashes, counts

The judge is the run's own validator role (model, revision and generation
settings) from its resolved_config.yaml. The interaction files are only read.

    PYTHONPATH=src python scripts/judge_answer_pairs.py \\
        --interactions results/collection/<run>/train/interactions.jsonl \\
                       results/collection/<run>/validation/interactions.jsonl \\
        --output <artifacts>/<run>_judgments
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from mas_sae.agents.validator import Validator
from mas_sae.evaluation.behavior_labels import POLICIES, judge_answer_pairs
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from mas_sae.experiments.provenance import role_metadata
from mas_sae.experiments.records import read_jsonl, write_jsonl
from mas_sae.models.roles import configure_agent, load_role_models, resolve_roles

CANDIDATE_POLICIES = sorted(name for name, policy in POLICIES.items() if policy.candidate_rules is not None)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactions", type=Path, nargs="+", required=True,
                        help="interactions.jsonl files of one run")
    parser.add_argument("--run-config", type=Path,
                        help="the run's resolved_config.yaml (default: next to the first interactions file)")
    parser.add_argument("--label-policy", choices=CANDIDATE_POLICIES, default="v0.2.0-candidate",
                        help="policy whose labels decide which pairs need a verdict")
    parser.add_argument("--output", type=Path, required=True, help="new folder for the judgments")
    args = parser.parse_args()

    ensure_output_available(args.output)
    policy = POLICIES[args.label_policy]
    run_config = args.run_config or args.interactions[0].parent / "resolved_config.yaml"
    hashes_before = {str(path): sha256_file(path) for path in args.interactions}
    rows = [row for path in args.interactions for row in read_jsonl(path)]

    spec = resolve_roles(yaml.safe_load(run_config.read_text()))["validator"]
    validator = Validator(*load_role_models({"validator": spec})["validator"])
    configure_agent(validator, spec)

    def judge(question, answer_1, answer_2):
        verdict = validator.judge_same_answer(question, answer_1, answer_2)
        return verdict.relation, verdict.raw_output, validator.last_generation

    records = judge_answer_pairs(rows, policy, judge)
    if {str(path): sha256_file(path) for path in args.interactions} != hashes_before:
        raise RuntimeError("An interactions file changed while it was being judged")

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "judgments.jsonl", records)
    counts = Counter(f"{r['pair']} / {r['relation']}" for r in records)
    provenance = {
        "git_commit": get_git_commit(),
        "uncommitted_changes_sha256": get_git_diff_sha256(),
        "label_policy": {"name": args.label_policy, "version": policy.version,
                         "definition_sha256": policy.definition_sha256()},
        "judge": role_metadata(validator),
        "prompt_sha256": sha256_text(Validator.SAME_ANSWER_PROMPT),
        "run_config": str(run_config),
        "interactions_sha256": hashes_before,
        "episodes": len(rows),
        "judgments": len(records),
        "judgments_by_pair_and_verdict": dict(sorted(counts.items())),
    }
    write_json_atomic(args.output / "provenance.json", provenance)
    print(json.dumps({k: provenance[k] for k in ("episodes", "judgments", "judgments_by_pair_and_verdict")},
                     indent=2))


if __name__ == "__main__":
    main()
