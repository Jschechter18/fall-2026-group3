# Setup

Everything needed to run the project: the Python environment, Claude on Amazon Bedrock, Hugging
Face model access, S3 storage and Jupyter. The [README](../README.md) has the short version and the
experimental pipeline.

## Getting Started

### 1. Clone the Repository

```bash
git clone <git@github.com:Jschechter18/fall-2026-group3.git>
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

## Set Up Claude Access Through Amazon Bedrock

The Claude examples use AWS credentials for Amazon Bedrock, rather than an
Anthropic API key. Install or update the Conda environment first; it includes
`boto3` and `python-dotenv`.

### Get temporary AWS credentials

1. Sign in to the school AWS access portal using the URL provided by the professor
   or school administrator. Ask them for the URL if you do not have it.
2. Expand your assigned AWS account and locate the role authorized to use Bedrock.
3. Select **Access keys** (sometimes labeled **Command line or programmatic access**)
   beside that role.
4. In the manual credentials section, copy the **access key ID**, **secret access
   key**, and **session token** from the same session.

See the [AWS instructions for obtaining temporary credentials](https://docs.aws.amazon.com/singlesignon/latest/userguide/howtogetcredentials.html#how-to-get-temp-credentials-manual).
If the required account or role is missing, contact the professor or school
administrator. S3 access alone does not establish permission to invoke Bedrock.

### Create your local `.env` file

Create a file named `.env` in the repository root, alongside `environment.yml`.
If it already exists, add or update these entries without removing other settings:

```dotenv
aws_access_key_id=YOUR_ACCESS_KEY_ID
aws_secret_access_key=YOUR_SECRET_ACCESS_KEY
aws_session_token=YOUR_SESSION_TOKEN
```

Use these lowercase variable names exactly: the current example reads them with
`os.getenv()`. Paste only the credential values, without shell `export` commands
or a `[BedRock_LLM_API]` section header. No `CONFIG_FILE` setting is needed for
this example. `load_dotenv()` reads the file; it does not create it.

The file is ignored by Git. Keep real credentials out of tracked files, including
`example.env`, and do not share them in messages or screenshots. Temporary
credentials expire; obtain a fresh set and replace all three values when needed,
then restart the script or notebook session.

### Run the example

From the repository root with the `capstone` environment activated and the project
installed, run:

```bash
python examples/claude/ask_claude_api.py
```

This makes a real, billable Bedrock request. An expired-token error means the
temporary credentials need refreshing. An `AccessDenied` error may require the
professor or administrator to grant the role access to the requested Bedrock
model or inference profile.

## Model Access

Models used by the Solver-Critic pipeline (all from Hugging Face):

- **Baseline / current Solver (all checked-in runnable configs):**
  `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`
  (the validated snapshot recovered during the team's provenance review).
- **Cross-model Natural Critic used in protocol validation:**
  `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- **Production roles (`configs/collection/v2/natural_4b_full_*.yaml`):**
  Gemma-3-4B-it as Solver and Validator, Qwen3-4B-Instruct-2507 as Natural
  Critic, at the revisions above. Gemma-3-12B in BF16 does not fit one A10G.
- **Final collection (`configs/collection/distributed_final.yaml`, run `mixed_final_001`):**
  the same three roles at the same revisions.

Setup (once per machine):

1. Create or log in to a Hugging Face account: https://huggingface.co/
2. Accept the Gemma license at https://huggingface.co/google/gemma-3-4b-it
   (Gemma is gated; Qwen is not).
3. Create a **read** token at https://huggingface.co/settings/tokens.
   Never commit or share the token.
4. In the `capstone` environment run `hf auth login` and paste the token,
   then confirm with `hf auth whoami`.
5. Weights download on first use and are cached under `checkpoints/huggingface/`.

Rules so runs stay comparable across machines:

- Put the exact model `id` **and** `revision` in the experiment config
  (`roles.solver`, `roles.critic`, `roles.validator`); the legacy `model.id`
  form is still accepted for V1 configs.
- Do not silently substitute a newer `main` revision; a run's
  `resolved_config.yaml` records the resolved revision of every role.
- Explicit `dtype` and `device`/`device_map` are required; there is no
  automatic quantization, dtype fallback, or placement change.

## Persistent Storage with Amazon S3

Generated activations and SAE checkpoints are not committed to Git. The team uses Amazon S3 to transfer these artifacts between compute instances and preserve them outside an individual EC2 instance.

Remote artifacts are stored under:

```text
s3://dats-capstone/shared/group3/
```

### Install the AWS CLI

On the course Ubuntu EC2 instance, install AWS CLI v2 for the current user:

```bash
curl -fsSL https://awscli.amazonaws.com/v2/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
hash -r
```

Verify the installation:

```bash
aws --version
```

See the [official AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for other platforms.

### Configure School SSO Access

The course account uses AWS IAM Identity Center. Do not use long-lived IAM access keys.

1. Sign in to the school AWS access portal.
2. Expand the assigned AWS account.
3. Find the student role.
4. Select **Access keys**.
5. Under **AWS IAM Identity Center credentials (Recommended)**, copy the:
   - SSO start URL
   - SSO region

Configure a named CLI profile:

```bash
aws configure sso --profile school --use-device-code
```

Use the following values when prompted:

```text
SSO session name: school
SSO start URL: <copy from the AWS access portal>
SSO region: us-east-1
SSO registration scopes: sso:account:access
Default client Region: us-east-1
CLI default output format: json
```

If asked whether to configure AWS skills and an AWS MCP server, select **No**. They are not required for S3 synchronization.

The device-login process displays a URL and code. Open the URL on a computer with a browser, enter the code, and authorize the school AWS session.

For subsequent sessions, renew authentication with:

```bash
aws sso login --profile school --use-device-code
```

Verify the active identity:

```bash
aws sts get-caller-identity --profile school
```

If this command reports expired credentials, rerun `aws sso login`.

### S3 Directory Layout

Local paths are mirrored beneath the group S3 prefix:

| Local path          | S3 prefix                                            |
| ------------------- | ---------------------------------------------------- |
| `artifacts/collection/<run>/` (distributed collections, incl. `mixed_final_001`) | `s3://dats-capstone/shared/group3/artifacts/collection/<run>/` |
| `data/activations/` | `s3://dats-capstone/shared/group3/data/activations/` |
| `results/sae/`      | `s3://dats-capstone/shared/group3/results/sae/`      |
| `results/causal_replay/` | `s3://dats-capstone/shared/group3/results/causal_replay/` |
| `results/collection/` (`interactions.jsonl`) | `s3://dats-capstone/shared/group3/results/collection/` |

The `artifacts/collection/` row follows the convention in
[the mixed collection runbook](runbooks/mixed_collection_pipeline.md). To get the final
collection, sync `s3://dats-capstone/shared/group3/artifacts/collection/mixed_final_001/` to
`artifacts/collection/mixed_final_001/`, with `--dryrun` first as below.

### Upload Activations

Always preview a synchronization first:

```bash
aws s3 sync \
  data/activations/ \
  s3://dats-capstone/shared/group3/data/activations/ \
  --dryrun \
  --profile school
```

If the preview is correct, upload the files:

```bash
aws s3 sync \
  data/activations/ \
  s3://dats-capstone/shared/group3/data/activations/ \
  --profile school
```

Verify the remote contents:

```bash
aws s3 ls \
  s3://dats-capstone/shared/group3/data/activations/ \
  --recursive \
  --profile school
```

### Download Activations

To restore activations from S3, reverse the source and destination:

```bash
aws s3 sync \
  s3://dats-capstone/shared/group3/data/activations/ \
  data/activations/ \
  --dryrun \
  --profile school
```

After reviewing the preview:

```bash
aws s3 sync \
  s3://dats-capstone/shared/group3/data/activations/ \
  data/activations/ \
  --profile school
```

### Upload an SAE Checkpoint Directory

Replace `<layer>` and `<run-id>` with the applicable experiment values:

```bash
aws s3 sync \
  results/sae/musique/runs/<layer>/<run-id>/checkpoints/ \
  s3://dats-capstone/shared/group3/results/sae/musique/runs/<layer>/<run-id>/checkpoints/ \
  --dryrun \
  --profile school
```

After reviewing the preview, remove `--dryrun`:

```bash
aws s3 sync \
  results/sae/musique/runs/<layer>/<run-id>/checkpoints/ \
  s3://dats-capstone/shared/group3/results/sae/musique/runs/<layer>/<run-id>/checkpoints/ \
  --profile school
```

To restore that checkpoint directory, reverse the paths:

```bash
aws s3 sync \
  s3://dats-capstone/shared/group3/results/sae/musique/runs/<layer>/<run-id>/checkpoints/ \
  results/sae/musique/runs/<layer>/<run-id>/checkpoints/ \
  --profile school
```

### Synchronization Safety

- Use `--dryrun` before uploads and downloads.
- Do not use `--delete` unless remote or local deletion is explicitly intended.
- `sync` skips matching files and transfers missing or changed files.
- Uploading to an existing S3 key can replace that object.
- S3 authentication does not change the local `ActivationStore`; experiments continue reading and writing local paths.
- Never commit or share AWS access keys, secret keys, session tokens, or screenshots containing them.
- Do not place AWS credentials in `data_sync.py`, configuration files, notebooks, or the repository.
- An `AccessDenied` error means authentication succeeded but the school role lacks permission for the requested bucket or prefix.

## Jupyter Notebooks

After environment is set up, when using a jupyter notebook, make sure to run the following command to ensure you can select the capstone environment inside the kernel:

```bash
python -m ipykernel install --user \
  --name capstone \
  --display-name "Python 3.11 (capstone)"
```
