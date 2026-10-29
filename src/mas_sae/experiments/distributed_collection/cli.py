"""Prepare, merge, and split local collection artifacts; AWS CLI transfers files."""
import argparse
from pathlib import Path

from mas_sae.experiments.artifacts import ensure_output_available, write_json_atomic
from mas_sae.experiments.collection_config import load_collection_config
from .core import merge_run, split_run
from .dataset import prepare_manifest


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Reserve intervention membership and divide all questions into work chunks.")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--num-chunks", type=int, required=True, help="Work chunks distributed across EC2; independent of checkpoint batch size.")
    prepare.add_argument("--output", type=Path, required=True, help="New shared manifest.json; outputs live beside it.")
    merge = commands.add_parser("merge", help="Merge all complete work chunks and reconstruct activation indices.")
    merge.add_argument("--manifest", type=Path, required=True)
    split = commands.add_parser("split", help="Freeze train/validation/test and export ordinary and intervention runs separately.")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--config", type=Path, required=True)
    return parser


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        ensure_output_available(args.output)
        config = load_collection_config(args.config)
        manifest = prepare_manifest(config, num_chunks=args.num_chunks)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output, manifest, overwrite=False)
        reserved = sum(r["partition"] == "intervention" for r in manifest["questions"])
        print(f"Prepared {len(manifest['questions'])} questions, including {reserved} intervention baselines, "
              f"in {len(manifest['chunks'])} work chunks: {args.output}")
    elif args.command == "merge":
        print(merge_run(args.manifest))
    else:
        print(split_run(args.manifest, load_collection_config(args.config)))
