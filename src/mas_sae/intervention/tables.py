"""The result tables, computed from labeled rows. See the analysis plan in ``run_intervention.py``.

"Flip": Claude labels the regenerated A2 as the opposite of the episode's stored label. Agreement
and out labels count as not flipped, and are reported as their own counts (``label_counts``).
Every comparison is paired by question; random flips are averaged over the draws within a question
first. Strata are pooled for Tables P, 2 and D: an arm's two strata are mirror images.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mas_sae.intervention import stats
from mas_sae.intervention.conditions import Condition
from mas_sae.intervention.edits import ARMS
from mas_sae.intervention.episodes import OPPOSITE, STRATA

ARM_NAMES = {("adoption", "decrease"): "decrease adoption features",
             ("adoption", "increase"): "increase retention features",
             ("retention", "decrease"): "decrease retention features",
             ("retention", "increase"): "increase adoption features"}


def arm_name(stratum: str, arm: str) -> str:
    return ARM_NAMES.get((stratum, arm), arm)


def stratum_name(stratum: str) -> str:
    return f"{stratum} → {OPPOSITE[stratum]}"


def ci_text(result: dict) -> str:
    lo, hi = result["ci95_pp"]
    return f"{result['delta_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


class Outcomes:
    """Flip outcomes and telemetry of labeled rows, looked up by (episode, condition)."""

    def __init__(self, rows: list[dict], edit_size: float, n_draws: int, plan):
        self.rows = {(r["episode_id"], r["condition"]): r for r in rows}
        self.stratum = {r["episode_id"]: r["stored_label"] for r in rows}
        self.edit_size, self.n_draws, self.plan = edit_size, n_draws, plan

    def episodes(self, stratum: str | None = None) -> list[str]:
        return sorted(e for e, s in self.stratum.items() if stratum is None or s == stratum)

    def flips(self, condition: Condition, episodes: list[str]) -> np.ndarray:
        rows = [self.rows[(e, condition.key)] for e in episodes]
        return np.array([r["label"] == OPPOSITE[r["stored_label"]] for r in rows], float)

    def _edit(self, rep, arm, source, draw=None) -> Condition:
        return Condition("edit", rep, arm, source, draw, self.edit_size)

    def targeted(self, rep: str, arm: str, episodes) -> np.ndarray:
        return self.flips(self._edit(rep, arm, "targeted"), episodes)

    def random(self, rep: str, arm: str, episodes) -> np.ndarray:
        """Per question, the mean flip over the random draws."""
        return np.mean([self.flips(self._edit(rep, arm, "random", k), episodes) for k in range(self.n_draws)], axis=0)

    def delta(self, rep: str, arm: str, episodes) -> np.ndarray:
        return self.targeted(rep, arm, episodes) - self.random(rep, arm, episodes)

    def logprob_shift(self, rep: str, arm: str, episodes) -> float:
        """Mean change in log p(baseline's first token), targeted − mean of random draws."""
        lp = lambda c: np.array([self.rows[(e, c.key)]["baseline_first_token_logprob"] for e in episodes], float)
        random = np.mean([lp(self._edit(rep, arm, "random", k)) for k in range(self.n_draws)], axis=0)
        return float(np.mean(lp(self._edit(rep, arm, "targeted")) - random))

    def test(self, a, b, alternative: str) -> dict:
        return stats.paired_difference(a, b, alternative, self.plan.n_bootstrap, self.plan.n_permutations, seed=0)


def with_holm(results: list[dict]) -> list[dict]:
    adjusted = stats.holm([r["p_value"] for r in results])
    return [{**r, "p_holm": p} for r, p in zip(results, adjusted)]


def table_p(o: Outcomes) -> pd.DataFrame:
    """PRIMARY: each pre-specified SAE × arm, strata pooled, one-sided, Holm across all 9."""
    episodes, tests = o.episodes(), []
    for rep in o.plan.primary_representations:
        for arm in ARMS:
            t, r = o.targeted(rep, arm, episodes), o.random(rep, arm, episodes)
            tests.append({"rep": rep, "arm": arm, "t": t.mean(), "r": r.mean(),
                          **o.test(t, r, o.plan.primary_alternative)})
    return pd.DataFrame([{
        "Representation": x["rep"], "Arm": x["arm"], "N": x["n"],
        "Targeted flip %": 100 * x["t"], "Random flip % (mean of draws)": 100 * x["r"],
        "Δ (pp) [95% CI]": ci_text(x), "p (one-sided)": x["p_value"], "p (Holm)": x["p_holm"],
        "Survives Holm (α = 0.05)": x["p_holm"] < 0.05} for x in with_holm(tests)])


def table_1(o: Outcomes, roles: dict) -> pd.DataFrame:
    """Per stratum. p (Holm) is corrected within each representation (its 6 rows)."""
    out = []
    for rep, role in roles.items():
        tests = []
        for stratum in STRATA:
            episodes = o.episodes(stratum)
            for arm in ARMS:
                t, r = o.targeted(rep, arm, episodes), o.random(rep, arm, episodes)
                tests.append({"stratum": stratum, "arm": arm, "t": t.mean(), "r": r.mean(),
                              "lp": o.logprob_shift(rep, arm, episodes),
                              **o.test(t, r, o.plan.secondary_alternative)})
        out += [{"Representation": rep + (" †" if role == "exploratory" else ""),
                 "Stratum (baseline → target)": stratum_name(x["stratum"]), "Arm": arm_name(x["stratum"], x["arm"]),
                 "N": x["n"], "Targeted flip %": 100 * x["t"], "Random flip % (mean of draws)": 100 * x["r"],
                 "Δ (pp) [95% CI]": ci_text(x), "p": x["p_value"], "p (Holm)": x["p_holm"],
                 "Δ log p(baseline first token)": x["lp"]} for x in with_holm(tests)]
    return pd.DataFrame(out)


def reference_rows(o: Outcomes, saes: list[str]) -> pd.DataFrame:
    """No-edit conditions: descriptive, no tests."""
    out = []
    for stratum in STRATA:
        episodes = o.episodes(stratum)
        out.append({"Condition": "Baseline (no SAE, no edit)", "Stratum": stratum_name(stratum), "N": len(episodes),
                    "Flip %": 100 * o.flips(Condition("baseline"), episodes).mean(),
                    "Meaning": "should be ~0%: the replay reproduces the stored labels"})
        for sae in saes:
            out.append({"Condition": f"{sae} spliced in, no edit", "Stratum": stratum_name(stratum), "N": len(episodes),
                        "Flip %": 100 * o.flips(Condition("reconstruction", sae), episodes).mean(),
                        "Meaning": "flips caused only by the SAE's reconstruction error"})
    return pd.DataFrame(out)


def table_2(o: Outcomes, pairs, family: str) -> pd.DataFrame:
    """Between-SAE contrasts, strata pooled, one row per arm: Δ_A − Δ_B, Holm within the family."""
    episodes, tests = o.episodes(), []
    for a, b in pairs:
        for arm in ARMS:
            tests.append({"a": a, "b": b, "arm": arm,
                          **o.test(o.delta(a, arm, episodes), o.delta(b, arm, episodes), o.plan.secondary_alternative)})
    return pd.DataFrame([{"Family": family, "Contrast (A vs. B)": f"{x['a']} vs. {x['b']}", "Arm": x["arm"],
                          "N": x["n"], "Δ_A − Δ_B (pp) [95% CI]": ci_text(x), "p": x["p_value"],
                          "p (Holm)": x["p_holm"]} for x in with_holm(tests)])


def table_d(o: Outcomes, representations: list[str]) -> pd.DataFrame:
    """Which direction works best: within each representation, the arms' Δs compared, Holm across its contrasts."""
    episodes, out = o.episodes(), []
    for rep in representations:
        tests = [{"a": a, "b": b, **o.test(o.delta(rep, a, episodes), o.delta(rep, b, episodes),
                                           o.plan.secondary_alternative)}
                 for a, b in o.plan.direction_contrasts]
        out += [{"Representation": rep, "Contrast": f"{x['a']} − {x['b']}", "N": x["n"],
                 "Δ difference (pp) [95% CI]": ci_text(x), "p": x["p_value"], "p (Holm)": x["p_holm"]}
                for x in with_holm(tests)]
    return pd.DataFrame(out)


def table_3(rows: list[dict], set_cosines: dict) -> pd.DataFrame:
    """Diagnostics: did the edits do what was intended? Descriptive."""
    df = pd.DataFrame([r for r in rows if r["kind"] == "edit"])
    recon = pd.DataFrame([r for r in rows if r["kind"] == "reconstruction"])
    out = []
    for rep, group in df.groupby("representation", sort=False):
        for name, part in (("adoption (targeted)", group[(group.source == "targeted") & (group.edited_set == "adoption")]),
                           ("retention (targeted)", group[(group.source == "targeted") & (group.edited_set == "retention")]),
                           ("random draws", group[(group.source == "random") & (group.edited_set != "both")])):
            size = part.applied_relative_size
            firing = part.edited_features_active.dropna()
            rec = recon[recon.representation == rep].reconstruction_error if len(recon) else pd.Series(dtype=float)
            out.append({"Representation": rep, "Feature set": name,
                        "Applied ‖Δ‖/‖x‖ mean [min, max]": f"{size.mean():.4f} [{size.min():.4f}, {size.max():.4f}]",
                        "% with an edited feature firing": None if firing.empty else 100 * float((firing > 0).mean()),
                        "Recon error (rel. to ‖x‖)": None if rec.empty else float(rec.mean()),
                        "cos(u_adoption, u_retention)": set_cosines[rep]})
    return pd.DataFrame(out)


def label_counts(rows: list[dict]) -> pd.DataFrame:
    """How often each label occurs per representation × arm × feature source (agreement and out included)."""
    df = pd.DataFrame(rows)
    df["representation"] = df.representation.fillna("none")
    for k in ("arm", "source"):
        df[k] = df[k].fillna("—")
    counts = df.groupby(["kind", "representation", "arm", "source", "label"], sort=False).size().unstack(fill_value=0)
    return counts.reindex(columns=["adoption", "retention", "agreement", "out"], fill_value=0).reset_index()


def to_markdown(df: pd.DataFrame) -> str:
    """A plain markdown table (floats to 4 significant digits)."""
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (
        f"{v:.4g}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"
