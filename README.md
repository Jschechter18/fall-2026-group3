# Capstone Project

Master's Data Science Capstone project.

## Getting Started

### 1. Clone the Repository

```bash
git clone <fall-2026-group3-repo-url>
cd fall-2026-group3
```

### 2. Install Conda

This project uses Conda to manage the Python environment and dependencies.

If Conda is not already installed, install either **Miniconda** or **Anaconda** before continuing.

### 3. Create the Environment

The project's dependencies are defined in `environment.yml`.

From the root directory of the repository, run:

```bash
conda env create -f environment.yml
```

This will create the `capstone` environment with the appropriate Python version and project dependencies.

### 4. Activate the Environment

```bash
conda activate capstone
```

### 5. Install the project

```bash
python -m pip install -e .
```

Verify that the environment is using Python 3.11:

```bash
python --version
```

You should see:

```text
Python 3.11.x
```

### Updating the Environment

If `environment.yml` changes after you have already created the environment, update your local environment with:

```bash
conda env update -f environment.yml --prune
```

This installs new dependencies and removes dependencies that are no longer specified in the environment file.

## Project Structure

```text
.
├── cookbooks/                    # Jupyter notebooks and tutorials
├── data/                         # Downloaded project datasets
├── demo/                         # Demonstrations and demo figures
├── documents/                    # Supporting project documents and references
├── presentation/                 # Presentation materials
├── reports/                      # Project and progress reports
├── research_paper/               # Research paper source and materials
├── results/                      # Experimental outputs and results
├── scripts/                      # Executable Python entry-point scripts
├── src/
│   ├── mas_sae/                 # Reusable Python package
│   │   ├── agents/              # Solver and critic agent logic
│   │   ├── data/                # Dataset loading and processing
│   │   ├── evaluation/          # Evaluation functions and metrics
│   │   ├── models/              # Language-model loading and interaction
│   │   └── sae/                 # Sparse-autoencoder functionality
│   └── tests/                   # Unit and integration tests
├── environment.yml               # Conda environment and dependencies
├── pyproject.toml                # Python package configuration
├── pytest.ini                    # Pytest configuration
└── README.md                     # Setup and project documentation
```

The project structure may change as development progresses.

## Development

When adding a new dependency, add it to `environment.yml` so that all team members use a consistent environment.

After modifying `environment.yml`, update your environment:

```bash
conda env update -f environment.yml --prune
```

## Jupyter Notebooks

After environment is set up, when using a jupyter notebook, make sure to run the following command to ensure you can select the capstone environment inside the kernel:

```bash
python -m ipykernel install --user \
  --name capstone \
  --display-name "Python 3.11 (capstone)"
```

## Data Collection

After activating the Conda environment and installing the project, run the following command from the repository root:

```bash
python scripts/get_musique_dataset.py
```

## Testing

After activating the Conda environment and installing the project, run:

```bash
pytest
```
