from typing import Any

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def load_gemma(
    model_id: str = "google/gemma-3-4b-it",
    cache_dir: str = "checkpoints/huggingface",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Gemma3ForConditionalGeneration, Any]:
    """Load the Gemma model and processor used by the agent pipeline."""
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=dtype,
        device_map="auto",
    )

    model.eval()

    return model, processor
