"""Achieved full-ablation edit sizes ||Δx||/||x|| before a causal run (#220): the targeted set and each of the
replay's own contribution-matched random draws, per SAE, on a seeded sample of validation activations.

Δ is the replay's error-preserving ablation edit, decoder(z with S zeroed) - decoder(z) (SAESplice._edit), and the
draws come from replay.feature_sets, so the sets are exactly the ones a run of that config generates. It reports
only; nothing here changes a scientific parameter. The TopK config calibrates the audit against the measured
diagnostics of its finished run. Run from the repo root with the configs' SAE files and the layer-25 activations
in place:
    python scripts/causal_edit_norm_audit.py results/causal_pretrained_layer25/edit_norm_audit.json
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mas_sae.causal import replay
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.probe.feature_selection import is_activation_basis, load_sae, sae_files

spec = importlib.util.spec_from_file_location("ft", "scripts/sae/pretrained-tuning/finetune_pretrained_sae.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

root = Path(".")
X = ft.ActivationStore(Path("artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25")) \
    .load_activations("validation_attempt2").float()
rows = np.random.default_rng(42).choice(len(X), size=1000, replace=False)
X = X[np.sort(rows)]
x_norm = X.norm(dim=1)


@torch.no_grad()
def ablation_sizes(sae, features, z, recon):
    edited = z.clone()
    edited[:, features] = 0
    return ((sae.decoder(edited) - recon).norm(dim=1) / x_norm).numpy()


def audit(config):
    cfg = replay.ReplayConfig.from_yaml(config)
    hashes = {n: {str(p.relative_to(root)): sha256_file(p) for p in sae_files(s, root)} for n, s in cfg.saes.items()}
    out = {}
    for name, f in replay.feature_sets(cfg, hashes).items():
        sae = load_sae(cfg.saes[name], root, cfg.layer)[0].eval()
        assert not is_activation_basis(sae)
        with torch.no_grad():
            z = sae.encoder(X)
            recon = sae.decoder(z)
        for label, g in [("all 10", f), *f.get("subsets", {}).items()]:
            t = ablation_sizes(sae, g["targeted"], z, recon)
            r = np.stack([ablation_sizes(sae, d, z, recon) for d in g["random"]], axis=1)    # rows x draws
            rmean = r.mean(axis=1)
            res = {"targeted_mean": float(t.mean()), "random_mean": float(r.mean()),
                   "targeted_median": float(np.median(t)), "random_median": float(np.median(r)),
                   "targeted_iqr": [float(q) for q in np.percentile(t, [25, 75])],
                   "random_iqr": [float(q) for q in np.percentile(r, [25, 75])],
                   "ratio_of_means": float(t.mean() / r.mean()) if r.mean() > 0 else None,
                   "random_draw_means_5_95": [float(q) for q in np.percentile(r.mean(axis=0), [5, 95])],
                   "targeted_noop_pct": float(100 * (t == 0).mean()), "random_noop_pct": float(100 * (r == 0).mean()),
                   # a 1%/2% dose is capped at ablation where the ablation edit is smaller than the dose
                   **{f"capped_at_{d}%_pct": {"targeted": float(100 * ((t > 0) & (t < d / 100)).mean()),
                                              "random": float(100 * ((r > 0) & (r < d / 100)).mean())} for d in (1, 2)},
                   "rows_targeted_below_random_mean_pct": float(100 * (t < rmean).mean()),
                   "n_rows": len(t), "n_draws": r.shape[1]}
            out[f"{name} · {label}"] = res
            print(f"{name + ' · ' + label:34s} targeted {100 * res['targeted_mean']:6.2f}%  random {100 * res['random_mean']:6.2f}%"
                  f"  ratio {res['ratio_of_means']:6.2f}×  capped@2% targeted {res['capped_at_2%_pct']['targeted']:5.1f}%"
                  f" random {res['capped_at_2%_pct']['random']:5.1f}%  no-op t/r {res['targeted_noop_pct']:.1f}/{res['random_noop_pct']:.1f}%")
    return out


ACT = Path("artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25/validation_attempt2.pt")
CONFIGS = ("configs/causal_topk_mixed_layer25.yaml", "configs/causal_pretrained_layer25.yaml")
result = {"provenance": {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         "uncommitted_code": subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], text=True).strip(),
                         "activations_sha256": sha256_file(ACT), "config_sha256": {c: sha256_file(Path(c)) for c in CONFIGS},
                         "rows_seed": 42, "n_rows": 1000},
          "population": "1,000 seeded validation rows (layer 25, attempt 2); the replay edits intervention episodes",
          "TopK (calibration, causal_topk_mixed_layer25.yaml)": audit("configs/causal_topk_mixed_layer25.yaml"),
          "Run A (causal_pretrained_layer25.yaml)": audit("configs/causal_pretrained_layer25.yaml")}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
