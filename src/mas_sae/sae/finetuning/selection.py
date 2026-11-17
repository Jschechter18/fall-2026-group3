"""Checkpoint eligibility: an epoch can be kept only if the fine-tune stayed light."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def within_l0(l0: float, target: float, tolerance: float) -> bool:
    """True if a checkpoint's mean L0 is within a relative tolerance of the original's (target)."""
    return abs(l0 - target) <= tolerance * target


def stability_limits(args: argparse.Namespace) -> dict[str, float]:
    """The three stability gates from the command line, as recorded in summary.json."""
    return {'min_activation_correlation': args.min_activation_correlation, 'min_jaccard': args.min_jaccard,
            'max_alive_change': args.max_alive_change}


def eligibility(l0: float, health: dict, target_l0: float,
                args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Can this epoch be kept? Returns (eligible, reasons it is not).

    The fine-tune should be light, so beyond lowering MSE an epoch must keep:
      - validation L0 within l0_tolerance of the original;
      - the median activation correlation with the original (features alive in both) at least
        min_activation_correlation (0 disables);
      - the mean active-set Jaccard with the original at least min_jaccard (0 disables);
      - the number of features alive on validation within max_alive_change of the original (inf disables).
    `health` is the epoch's compact health/stability block.
    """
    reasons = []
    if not within_l0(l0, target_l0, args.l0_tolerance):
        reasons.append(f'L0 {l0:.4g} not within {args.l0_tolerance:g} of {target_l0:.4g}')
    correlation = health['activation_correlation_median']
    if args.min_activation_correlation > 0 and (correlation is None or correlation < args.min_activation_correlation):
        reasons.append(f'activation correlation median {_format(correlation)} < {args.min_activation_correlation:g}')
    jaccard = health['active_set_jaccard_mean']
    if args.min_jaccard > 0 and jaccard < args.min_jaccard:
        reasons.append(f'active-set Jaccard mean {jaccard:.4g} < {args.min_jaccard:g}')
    change = health['alive_relative_change']
    if change is not None and abs(change) > args.max_alive_change:
        reasons.append(f'alive features changed by {change:+.2%} (limit {args.max_alive_change:g})')
    return not reasons, reasons


def _format(value):
    return 'n/a' if value is None else f'{value:.4g}'
