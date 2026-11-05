"""The paired tests, Holm, and the tables on synthetic labeled rows with a known effect."""

import numpy as np
import pytest

from mas_sae.intervention.analysis import stats, tables
from mas_sae.intervention.experiment.conditions import Condition
from mas_sae.intervention.experiment.config import Analysis
from mas_sae.intervention.generation.edits import ARMS
from mas_sae.intervention.experiment.episodes import OPPOSITE
from mas_sae.intervention.analysis.pilot import is_degenerate, longest_repeat_run

SIZE, DRAWS = 0.05, 3
PLAN = Analysis(primary_representations=("Causal SAE", "Null SAE"),
                between_sae_contrasts=(("Causal SAE", "Null SAE"),), exploratory_contrasts=(),
                direction_contrasts=(("both", "decrease"), ("both", "increase"), ("increase", "decrease")),
                n_bootstrap=2000, n_permutations=2000)


def test_identical_values_have_no_effect():
    a = np.random.default_rng(0).integers(0, 2, 200).astype(float)
    r = stats.paired_difference(a, a, "two-sided", 500, 500, seed=0)
    assert r["delta_pp"] == 0 and r["p_value"] == 1.0 and r["ci95_pp"] == [0, 0]


def test_a_real_effect_is_detected_one_sided_and_its_reverse_is_not():
    rng = np.random.default_rng(0)
    targeted, random = (rng.random(300) < 0.4).astype(float), (rng.random(300) < 0.1).astype(float)
    up = stats.paired_difference(targeted, random, "greater", 1000, 2000, seed=0)
    assert up["delta_pp"] == pytest.approx(100 * (targeted - random).mean())
    assert up["ci95_pp"][0] > 0 and up["p_value"] < 0.001
    assert stats.paired_difference(random, targeted, "greater", 1000, 2000, seed=0)["p_value"] > 0.99


def test_paired_values_must_line_up():
    with pytest.raises(ValueError, match="one value of each"):
        stats.paired_difference([1.0, 0.0], [1.0], "greater", 10, 10, seed=0)


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def _labeled_rows(n=300, seed=0):
    """'Causal SAE' targeted edits flip 50% of episodes and random ones 5%; 'Null SAE' flips 5% either way."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        stored = "adoption" if i % 2 else "retention"
        def row(condition, p_flip):
            flip = rng.random() < p_flip
            label = OPPOSITE[stored] if flip else rng.choice([stored, "agreement", "out"])
            return {"episode_id": f"e{i}", "stored_label": stored, "condition": condition.key, **condition.as_dict(),
                    "label": label, "baseline_first_token_logprob": -1.0 - flip}
        rows.append(row(Condition("baseline"), 0.0))
        for rep, p_targeted in (("Causal SAE", 0.5), ("Null SAE", 0.05)):
            rows.append(row(Condition("reconstruction", rep), 0.0))
            for arm in ARMS:
                rows.append(row(Condition("edit", rep, arm, "targeted", None, SIZE), p_targeted))
                rows += [row(Condition("edit", rep, arm, "random", k, SIZE), 0.05) for k in range(DRAWS)]
    return rows


def test_tables_find_the_planted_effect_and_only_it():
    o = tables.Outcomes(_labeled_rows(), SIZE, DRAWS, PLAN)
    p = tables.table_p(o)
    causal, null = p[p.Representation == "Causal SAE"], p[p.Representation == "Null SAE"]
    assert len(p) == 6 and causal["Survives Holm (α = 0.05)"].all() and not null["Survives Holm (α = 0.05)"].any()
    assert (p["p (Holm)"] >= p["p (one-sided)"]).all()
    assert causal["Targeted flip %"].between(40, 60).all() and causal["Random flip % (mean of draws)"].between(1, 10).all()

    t1 = tables.table_1(o, {"Causal SAE": "primary", "Null SAE": "exploratory"})
    assert len(t1) == 12 and set(t1.Representation) == {"Causal SAE", "Null SAE †"}
    assert set(t1.Arm) == {"decrease adoption features", "increase retention features", "both",
                           "decrease retention features", "increase adoption features"}

    t2 = tables.table_2(o, PLAN.between_sae_contrasts, "pre-specified")
    assert len(t2) == 3 and all(s.startswith("+") for s in t2["Δ_A − Δ_B (pp) [95% CI]"])
    assert (t2["p (Holm)"] < 0.05).all()

    d = tables.table_d(o, ["Causal SAE"])
    assert len(d) == 3 and (d["p (Holm)"] > 0.05).all()            # every arm was given the same effect


def test_agreement_and_out_count_as_not_flipped_and_are_counted():
    rows = _labeled_rows(n=40)
    o = tables.Outcomes(rows, SIZE, DRAWS, PLAN)
    baseline_flips = o.flips(Condition("baseline"), o.episodes())
    assert baseline_flips.sum() == 0                               # baseline rows are stored, agreement or out
    counts = tables.label_counts(rows)
    assert list(counts.columns[-4:]) == ["adoption", "retention", "agreement", "out"]
    assert counts[["adoption", "retention", "agreement", "out"]].to_numpy().sum() == len(rows)


def test_reference_rows_and_markdown():
    o = tables.Outcomes(_labeled_rows(n=20), SIZE, DRAWS, PLAN)
    ref = tables.reference_rows(o, ["Causal SAE", "Null SAE"])
    assert len(ref) == 2 * 3 and (ref["Flip %"] == 0).all()
    md = tables.to_markdown(ref)
    assert md.startswith("| Condition | Stratum |") and md.count("\n") == len(ref) + 2


def test_degenerate_answers():
    ok = {"answer": "Paris", "reached_token_budget": False, "generated_token_ids": [5, 6, 1]}
    assert not is_degenerate(ok)
    assert is_degenerate({**ok, "answer": "  "})
    assert is_degenerate({**ok, "reached_token_budget": True})
    assert is_degenerate({**ok, "generated_token_ids": [3, 7, 7, 7, 7, 7, 2]})
    assert longest_repeat_run([]) == 0 and longest_repeat_run([1, 1, 2, 2, 2]) == 3
