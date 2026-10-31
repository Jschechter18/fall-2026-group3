import json
from dataclasses import dataclass

from boto3.session import Session
from botocore.config import Config
from typing import Literal


@dataclass(frozen=True)
class ClaudeResult:
    """Claude response text, model identifier, token usage, and stop reason."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    task: str
    stop_reason: str | None
    sdk_retry_attempts: int = 0

class ClaudeClient:
    """Reusable client for calling Claude through Amazon Bedrock."""
    def __init__(self, session: Session, *, model_id: str, region: str = "us-east-1",
                 config: Config | None = None):
        """Initialize the ClaudeClient with a Boto3 session and model details.

        Parameters
        ----------
        session : Session
            Boto3 session object used to create the Bedrock client.
        model_id : str
            The model ID for the Claude model to be used.
        region : str, optional
            The AWS region where the Bedrock client is created, by default "us-east-1"
        config : Config | None, optional
            Optional SDK transport timeout and retry configuration.
        """
        self.model_id = model_id
        self._client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=config,
        )
        
    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 4096,
                 temperature: float | None = None, task: Literal["data-gen", "labeling", "other"]) -> ClaudeResult:
        """Send a prompt to Claude and return its response and usage metadata.

        Parameters
        ----------
        prompt : str
            User message containing the task input.
        task : {"data-gen", "labeling", "other"}
            Description of the task to be performed by Claude. Must be one of "data-gen", "labeling", or "other".
        system : str, optional
            Instructions defining Claude's role and behavior, by default "".
        max_tokens : int, optional
            Maximum number of output tokens, by default 4096.
            Responses may finish before reaching this limit.
        temperature : float, None, optional
            None omits the parameter from the API request.
            Controls randomnes in generation. Lower values favor more consistent responses.
            This is especially useful for controlling the creativity of the model's responses, consider with data generation.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if temperature is not None and not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1.")
        if task not in ("data-gen", "labeling", "other"):
            raise ValueError("task must be 'data-gen', 'labeling', or 'other'.")
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        
        response = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        stream = response["body"]
        try:
            data = json.loads(stream.read())
        finally:
            stream.close()
            
        return ClaudeResult(
            text="".join(
                block["text"]
                for block in data["content"]
                if block["type"] == "text"
            ),
            model=data["model"],
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            task=task,
            stop_reason=data.get("stop_reason"),
            sdk_retry_attempts=response.get("ResponseMetadata", {}).get("RetryAttempts", 0),
        )

    
    def close(self):
        """Close the underlying Bedrock client's network connections.
        """
        self._client.close()
    