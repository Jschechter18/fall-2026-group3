import boto3

import os
from dotenv import load_dotenv  #  pip3 install python-dotenv

from mas_sae.models.claude import ClaudeClient

load_dotenv()


session = boto3.Session(
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key'),
    aws_session_token=os.getenv('aws_session_token')
)


claude = ClaudeClient(
    session=session,
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)

prompt = "What is the capital of France?"

try:
    result = claude.generate(prompt, task="other")
    print(result.text)
    print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
finally:
    claude.close()
