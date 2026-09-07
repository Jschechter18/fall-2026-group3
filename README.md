# Capstone Project

Master's Data Science Capstone project.

## Getting Started

### 1. Clone the Repository

```bash
git clone <agent-evidence>
cd <agent-evidence>
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
├── environment.yml       # Conda environment and dependencies
├── README.md             # Project documentation
├── data/                 # Project datasets
├── notebookc/            # Exploratory analysis and experiments
├── Code/src                  # Source code
└── results/              # Experimental outputs and results
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

Before continuing with development or using the repo, make sure to run the data collection script to ensure you have the data in the repo.

After activating the codna environment, and before using the code, run the following:

```bash
python get_MuSiQue_dataset.py
```

If using
