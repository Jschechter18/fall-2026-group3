from unittest.mock import Mock, patch

import torch

from mas_sae.models.loader import load_gemma


@patch("mas_sae.models.loader.Gemma3ForConditionalGeneration.from_pretrained")
@patch("mas_sae.models.loader.AutoProcessor.from_pretrained")
def test_load_gemma_uses_expected_configuration(
    mock_processor_from_pretrained,
    mock_model_from_pretrained,
):
    processor = Mock()
    model = Mock()

    mock_processor_from_pretrained.return_value = processor
    mock_model_from_pretrained.return_value = model

    loaded_model, loaded_processor = load_gemma(
        model_id="test-model",
        cache_dir="test-cache",
        dtype=torch.bfloat16,
    )

    mock_processor_from_pretrained.assert_called_once_with(
        "test-model",
        cache_dir="test-cache",
    )

    mock_model_from_pretrained.assert_called_once_with(
        "test-model",
        cache_dir="test-cache",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval.assert_called_once_with()

    assert loaded_model is model
    assert loaded_processor is processor
