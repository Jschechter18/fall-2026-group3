"""Local synthetic input uses the existing collection selection machinery."""
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from mas_sae.data import musique
from mas_sae.experiments import artifacts, collection, collection_progress, provenance
from mas_sae.experiments.collection_config import load_collection_config


@pytest.fixture
def local_dataset(tmp_path, monkeypatch):
    rows = [
        {"id": f"{hop}__synthetic_{i}", "question": f"Question {hop} {i}?",
         "answer": str(i), "paragraphs": [], "answerable": True}
        for hop in ("2hop", "3hop1", "4hop3") for i in range(10)
    ]
    rows.append({**rows[0], "id": "2hop__unanswerable", "answerable": False})
    path = tmp_path / "accepted.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    hub = Mock(side_effect=AssertionError("Local collection must not fetch MuSiQue"))
    monkeypatch.setattr(musique, "load_dataset", hub)
    return path, rows


@pytest.mark.parametrize("strategy", ["first_n", "random", "stratified", "listed"])
def test_local_selection_preserves_records_and_is_repeatable(local_dataset, tmp_path, strategy):
    path, rows = local_dataset
    sampling = {"strategy": strategy, "seed": 42}
    if strategy == "listed":
        ids_path = tmp_path / "ids.json"
        ids_path.write_text(json.dumps([row["id"] for row in rows[:6]][::-1]))
        sampling["question_ids_file"] = str(ids_path)
    if strategy == "stratified":
        sampling["hop_proportions"] = {"2hop": 1/3, "3hop": 1/3, "4hop": 1/3}
    config = {"path": str(path), "source_split": "train", "num_questions": 6,
              "sampling": sampling}
    result = collection.select_questions(config, default_seed=42, a2_presentations=["candidates"])
    assert result == collection.select_questions(config, default_seed=42, a2_presentations=["candidates"])
    assert len(result["examples"]) == 6
    assert all(row in rows and row["answerable"] for row in result["examples"])
    assert len(result["sampled_questions"]) == 6
    assert len(result["candidate_orders"]) == 6
    if strategy == "stratified":
        assert Counter(musique.hop_group(row["id"]) for row in result["examples"]) == {
            "2hop": 2, "3hop": 2, "4hop": 2}
    if strategy == "listed":
        assert [row["id"] for row in result["examples"]] == [row["id"] for row in rows[:6]][::-1]


def test_local_file_validation_and_short_pool(local_dataset):
    path, rows = local_dataset
    with pytest.raises(RuntimeError, match="only found"):
        musique.sample_musique_examples(source_split="train", num_questions=32, seed=42, path=path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        musique.load_musique_split("train", path=path, revision="a" * 40)
    for content, error in [("not JSON", "invalid JSON"), ("[]", "expected a MuSiQue"),
                           (json.dumps(rows[0]) + "\n" + json.dumps(rows[0]), "duplicate question id")]:
        path.write_text(content)
        with pytest.raises(ValueError, match=error):
            musique.load_musique_split("train", path=path)


def test_local_config_validation(local_dataset, tmp_path):
    path, _ = local_dataset
    config = yaml.safe_load(open("configs/collection/v2/smoke_train.yaml"))
    config["dataset"] = {"path": str(path), "source_split": "train", "num_questions": 6}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert load_collection_config(config_path)["dataset"]["path"] == str(path)
    for extra, error in [({"revision": "a" * 40}, "mutually exclusive"),
                         ({"path": ""}, "nonempty"), ({"path": str(tmp_path / "missing")}, "does not exist")]:
        invalid = {**config, "dataset": {**config["dataset"], **extra}}
        config_path.write_text(yaml.safe_dump(invalid))
        with pytest.raises(ValueError, match=error):
            load_collection_config(config_path)


def test_local_provenance_and_resume_detect_changed_content(local_dataset, tmp_path, monkeypatch):
    path, _ = local_dataset
    monkeypatch.setattr(provenance, "environment_metadata", lambda: {"gpus": []})
    monkeypatch.setattr(artifacts, "get_git_commit", lambda: "abc")
    monkeypatch.setattr(artifacts, "get_git_diff_sha256", lambda: "diff")
    monkeypatch.setattr(artifacts, "get_untracked_sha256", lambda *args: "untracked")
    monkeypatch.setattr(artifacts, "get_package_version", lambda *args: "version")
    config = {"dataset": {"path": str(path)}, "collection": {"seed": 42}}
    agent = SimpleNamespace(max_new_tokens=4)
    def identity():
        resolved = provenance.build_resolved_config(
            config, SimpleNamespace(config=SimpleNamespace()), agent, agent, agent)
        assert resolved["provenance"]["dataset_id"] == str(path.resolve())
        assert resolved["provenance"]["dataset_file_sha256"] == artifacts.sha256_file(path)
        return provenance.build_collection_progress_provenance(config, resolved, manifest_sha256="manifest")
    progress = tmp_path / "progress"
    collection_progress.initialize(progress, identity())
    collection_progress.check_identity(progress, identity())
    path.write_text(path.read_text().replace("Question", "Modified question"))
    with pytest.raises(RuntimeError, match="dataset_file_sha256"):
        collection_progress.check_identity(progress, identity())
