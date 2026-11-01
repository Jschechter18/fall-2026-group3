"""Run optional validation-only SAE refinement in isolated child directories."""
from pathlib import Path
from mas_sae.sae.refinement import refinement_cli
# Compatibility exports for historical experiment consumers.
from mas_sae.sae.refinement import ModelRunner, resolve_branches, measure_activity, save_checkpoint


def main():
    refinement_cli(Path(__file__).resolve().parents[2])


if __name__ == '__main__':
    main()
