"""Command-line interface for causal replay prepare, map, and reduce stages."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .distributed import prepare_distributed_run, reduce_shards, run_shard


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Freeze episode shards from a successful preflight run.")
    prepare.add_argument("--preflight-run", type=Path, required=True)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--output", type=Path, help="New distributed directory; defaults beside the preflight.")

    map_command = commands.add_parser("map", help="Run every intervention condition for one episode shard.")
    map_command.add_argument("--manifest", type=Path, required=True)
    map_command.add_argument("--shard", type=int, required=True)
    map_command.add_argument("--resume", action="store_true")

    reduce = commands.add_parser("reduce", help="Validate, merge, and analyse all completed shards.")
    reduce.add_argument("--manifest", type=Path, required=True)
    reduce.add_argument("--output", type=Path, help="New canonical run directory; defaults to distributed/merged.")
    return root


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "prepare":
        result = prepare_distributed_run(
            args.preflight_run, num_shards=args.num_shards, output_dir=args.output,
        )
    elif args.command == "map":
        result = run_shard(args.manifest, args.shard, resume=args.resume)
    else:
        result = reduce_shards(args.manifest, destination=args.output)
    print(result)
