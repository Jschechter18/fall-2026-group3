results/sae/musique/
├── tuning/
│ └── <study*id>/
│ ├── config.yaml # Search space, budget, selection rule
│ ├── trials.csv # Parameters, metrics, status, run paths
│ └── selection.json # Winning run and checkpoint path
│
└── runs/
└── layer*<N>/
├── sae_versions.csv
├── <trial_run_id>/
│ ├── config.json
│ ├── history.json
│ ├── ...existing run artifacts...
│ └── checkpoints/best_checkpoint.pt
└── <another_trial_run_id>/
