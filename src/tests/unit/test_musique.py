from pathlib import Path
from unittest.mock import Mock

from mas_sae.data import musique


def test_download_musique_writes_expected_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_split = Mock()
    validation_split = Mock()

    mock_load_dataset = Mock(
        return_value={
            "train": train_split,
            "validation": validation_split,
        }
    )
    monkeypatch.setattr(musique, "load_dataset", mock_load_dataset)

    output_dir = tmp_path / "MuSiQue" / "clean"

    train_path, validation_path = musique.download_musique(output_dir)

    expected_train_path = output_dir / "train.json"
    expected_validation_path = output_dir / "validation.json"

    mock_load_dataset.assert_called_once_with("dgslibisey/MuSiQue")
    train_split.to_json.assert_called_once_with(expected_train_path)
    validation_split.to_json.assert_called_once_with(expected_validation_path)

    assert output_dir.is_dir()
    assert train_path == expected_train_path
    assert validation_path == expected_validation_path