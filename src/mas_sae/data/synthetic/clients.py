"""Shared model default and Bedrock client construction."""
import os


DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5"


def make_client(model_id: str, *, connect_timeout: float = 10,
                read_timeout: float = 300, sdk_max_attempts: int = 3):
    """Create an independent session/client with bounded transport retries."""
    import boto3
    from botocore.config import Config
    from dotenv import load_dotenv
    from mas_sae.models.claude import ClaudeClient

    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    config = Config(connect_timeout=connect_timeout, read_timeout=read_timeout,
                    retries={"mode": "standard", "total_max_attempts": sdk_max_attempts})
    return ClaudeClient(session=session, model_id=model_id, config=config)
