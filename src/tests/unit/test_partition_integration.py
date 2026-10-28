"""Historical membership through distributed merging and real downstream loaders.

Synthetic inputs are handcrafted schema fixtures, not a generated release.
Models and Claude are not called; activation vectors encode question identity.
"""
import csv
import json
from pathlib import Path
import runpy

import pandas as pd
import pytest
import torch

from mas_sae.data.partition_adapter import import_historical_partitions
from mas_sae.data.production import (
    authorize_rows, build_partition_manifest, interaction_paths, load_labeled_rows,
    load_production_activations, read_manifest, read_partitions, read_run_interactions,
)
from mas_sae.evaluation.behavior import classify_behavior
from mas_sae.evaluation.behavior_labels import classify_candidate, write_labels, V020_CANDIDATE
from mas_sae.experiments import collection_progress
from mas_sae.experiments.distributed_collection import (
    build_master_dataset, freeze_partitions, plan_chunks, finalize_local_chunks,
)
from mas_sae.experiments.distributed_collection import cli
from mas_sae.experiments.records import read_jsonl
from mas_sae.probe import behavior_data as bd, layer_selection as ls
from mas_sae.sae.dataloader import create_sae_dataloader

PROPORTIONS = {"train": .5, "validation": .2, "test": .2, "intervention": .1}
SITE = "model.layers.8"


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    def question(qid):
        return {"id": qid, "question": "Where?", "paragraphs": [], "answer": "Paris"}
    train = [question(f"2hop__{i}_{i}") for i in range(1, 5)]
    val = [question("3hop__5_5")]
    synthetic = [{**question(qid), "hop_group": "2hop"} for qid in ("2hop__1_1", "new")]
    specs = []
    for name, source, split, rows in (("train", "musique", "train", train),
                                      ("val", "musique", "validation", val),
                                      ("synthetic", "synthetic", "", synthetic)):
        path = tmp_path/f"{name}.json"
        path.write_text(json.dumps(rows))
        specs.append({"source": source, "version": "fixture", "source_split": split,
                      "path": str(path), **({"hop_field": "hop_group"} if source == "synthetic" else {})})
    master = build_master_dataset(specs)
    historical = [{"question_id": row["id"], "source_split": "train", "partition": partition, "hop_group": "2hop"}
                  for row, partition in zip(train, ("train", "test", "intervention", "train"))]
    historical.append({"question_id": val[0]["id"], "source_split": "validation", "partition": "validation", "hop_group": "3hop"})
    path = tmp_path/"historical.csv"
    write_csv(path, historical)
    history = [{**r, "in_qc100": i == 0, "in_development_data": i == 0} for i, r in enumerate(historical)]
    exposure = tmp_path/"exposure.csv"
    write_csv(exposure, history)
    return master, path, exposure


def test_historical_import_preserves_membership_and_exposure(inputs):
    master, path, exposure = inputs
    unknown = import_historical_partitions(master, path)
    assert unknown["musique:2hop__1_1"]["in_development_data"] is None
    imported = import_historical_partitions(master, path, exposure_manifest=exposure)
    assert imported["musique:2hop__1_1"]["in_qc100"] is True
    assert imported["musique:2hop__1_1"]["in_development_data"] is True
    assert "synthetic:2hop__1_1" not in imported
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=42, existing=imported)
    by_id = {r["question_id"]: r for r in frozen["questions"]}
    assert all(by_id[q]["partition"] == r["partition"] for q, r in imported.items())
    assert by_id["musique:2hop__2_2"]["partition"] == "test"
    assert by_id["musique:2hop__3_3"]["partition"] == "intervention"


def test_historical_unknown_ids_and_origin_mismatches_fail(inputs):
    master, path, _ = inputs
    master["questions"] = [r for r in master["questions"] if r["question_id"] != "musique:2hop__1_1"]
    with pytest.raises(ValueError, match="missing from master"):
        import_historical_partitions(master, path)


def test_freeze_cli_accepts_historical_csv(inputs, tmp_path, monkeypatch):
    master, path, exposure = inputs
    (tmp_path/"master.json").write_text(json.dumps(master))
    (tmp_path/"config.json").write_text(json.dumps({"seed": 42, "proportions": PROPORTIONS}))
    output = tmp_path/"frozen.json"
    monkeypatch.setattr('sys.argv', ['distributed_collection.py', 'freeze', '--master', str(tmp_path/'master.json'),
        '--config', str(tmp_path/'config.json'), '--existing', str(path), '--exposure-manifest', str(exposure), '--output', str(output)])
    cli.main()
    rows = {r['question_id']: r for r in json.loads(output.read_text())['questions']}
    assert rows['musique:2hop__2_2']['partition'] == 'test'
    assert rows['musique:2hop__1_1']['in_development_data'] is True


@pytest.fixture
def collected(inputs, tmp_path):
    master, path, exposure = inputs
    old = import_historical_partitions(master, path, exposure_manifest=exposure)
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=42, existing=old)
    plan = plan_chunks(master, frozen, run_id="integration", chunk_size=1)
    numbers = {r["question_id"]: i+1 for i, r in enumerate(master["questions"])}
    excluded = "musique:2hop__4_4"
    locations = {}
    for chunk in plan["chunks"]:
        result = {"records": [], "exclusions": [], "attempt1_by_site": {SITE: []}, "attempt2_by_site": {SITE: []}}
        for assignment in chunk["questions"]:
            qid = assignment["question_id"]
            if qid == excluded:
                result["exclusions"].append({"question_id": qid, "seed": assignment["collection_seed"], "reason": "fixture exclusion"})
                continue
            value = numbers[qid]
            result["records"].append({"question_id": qid, "episode_id": qid+"__natural__candidates",
                "source_split": chunk["partition"], "seed": assignment["collection_seed"],
                "solver_accepted_feedback": True, "critic_condition": "natural", "a2_presentation": "candidates",
                "candidate_order": assignment["candidate_order"], "solver_attempt_1": "London",
                "critic_advocated_answer": "Paris", "solver_attempt_2": "Paris", "critic_noncommittal": False,
                "ground_truth": "Paris", "answer_aliases": [],
                "solver_attempt_1_correct": False, "critic_feedback_correct": True,
                "candidate_first_token_ids": {"a1": 1, "critic": 2},
                "decision_token_distinct": True, "solver_a2_first_token_id": 2,
                **classify_behavior("London", "Paris", "Paris", usable=True)})
            result["attempt1_by_site"][SITE].append(torch.tensor([[float(value), 1.]]))
            result["attempt2_by_site"][SITE].append(torch.tensor([[float(value+100), 2.]]))
        progress = tmp_path/"chunks"/chunk["chunk_id"]
        progress.mkdir(parents=True)
        locations[chunk["chunk_id"]] = collection_progress.write_chunk(progress, start=chunk["partition_start"],
            question_ids=chunk["question_ids"], candidate_sites=[SITE], result=result,
            metadata_extra={"chunk_id": chunk["chunk_id"], "run_id": plan["run_id"], "plan_sha256": plan["plan_sha256"],
                            "membership_sha256": chunk["membership_sha256"], "candidate_sites": [SITE]})
    finalize_local_chunks(plan=plan, chunk_dirs=locations, candidate_sites=[SITE], activation_root=tmp_path/"activations",
                          result_root=tmp_path/"results", run_name="merged", resolved_config={})
    return tmp_path/"results/merged", tmp_path/"activations/merged", numbers, frozen, excluded


def test_final_manifest_preserves_origin_membership_and_coverage(collected):
    results, activations, numbers, frozen, excluded = collected
    manifest = read_manifest(results/"split_manifest.csv")
    fixed = read_partitions(results/"partitions.csv", layout="partitioned")
    expected = {r["question_id"]: r["partition"] for r in frozen["questions"]}
    assert {q: r["partition"] for q, r in fixed.items()} == expected
    assert excluded not in manifest
    assert not (results/"intervention").exists()
    assert not list(activations.glob('*/intervention*.pt'))
    for qid, row in manifest.items():
        assert row["source_split"] == row["partition"] == expected[qid]
        assert row["hop_group"] in {"2hop", "3hop"}
        assert row["episode_id"] == qid+"__natural__candidates"
    assert manifest["musique:2hop__2_2"]["original_source_split"] == "train"
    assert manifest["musique:2hop__2_2"]["source_split"] == "test"
    assert manifest["musique:2hop__1_1"]["in_development_data"] is True
    assert fixed["musique:2hop__3_3"]["partition"] == "intervention"
    status = json.loads((results/"collection_status.json").read_text())
    assert status['reserved'] == ['musique:2hop__3_3']
    assert [r['question_id'] for r in status['excluded']] == [excluded]
    assert len(status['collected'])+len(status['excluded'])+len(status['reserved']) == len(expected)


def test_real_sae_and_production_activation_loaders(collected):
    results, activations, numbers, _, _ = collected
    manifest = read_manifest(results/"split_manifest.csv")
    rows = read_run_interactions(results, layout="partitioned")
    assert {r['source_split'] for r in rows} == {'train', 'validation', 'test'}
    for split, purpose in (("train", "fit"), ("validation", "tune"), ("test", "evaluate")):
        group = [r for r in rows if r['source_split'] == split]
        for attempt, offset in ((1, 0), (2, 100)):
            matrix = load_production_activations(group, manifest, activations, purpose=purpose, layer=8, attempt=attempt)
            assert matrix[:, 0].tolist() == [numbers[r['question_id']]+offset for r in group]
        combined = next(iter(create_sae_dataloader(100, split, 0, location=activations/'layer_08')))
        expected = [numbers[r['question_id']] for r in group]
        assert combined[:, 0].tolist() == expected+[v+100 for v in expected]
    with pytest.raises(ValueError, match="requires train"):
        authorize_rows([r for r in rows if r['source_split'] == 'test'], manifest, purpose='fit')
    with pytest.raises(ValueError, match="requires validation"):
        authorize_rows([r for r in rows if r['source_split'] == 'train'], manifest, purpose='tune')


def attach_fixture_labels(results):
    rows = read_run_interactions(results, layout='partitioned')
    for row in rows:
        row['label'] = classify_candidate(row, V020_CANDIDATE)
    write_labels(results/'labels.csv', rows)
    table = pd.read_csv(results/'labels.csv')
    episode_ids = {r['question_id']: r['episode_id'] for r in rows}
    table['episode_id'] = table.question_id.map(episode_ids)
    table.to_csv(results/'labels.csv', index=False)
    return rows


def test_label_join_and_real_probe_loaders(collected):
    results, activations, numbers, _, _ = collected
    attach_fixture_labels(results)
    rows, manifest = load_labeled_rows(results, results, layout='partitioned')
    assert {r['partition'] for r in rows} == {'train', 'validation', 'test'}
    selected, _ = load_labeled_rows(results, results, partitions={'test'}, layout='partitioned')
    assert all(r['partition'] == 'test' for r in selected)
    cols = bd.Columns()
    episodes = bd.build_episode_table(results, ['train', 'validation', 'test'], [])
    aligned, diagnostics = bd.align_episodes(episodes, pd.read_csv(results/'labels.csv'),
        pd.read_csv(results/'split_manifest.csv'), cols, 'full_index')
    assert diagnostics['labels_join'] == 'episode_id'
    with pytest.raises(PermissionError):
        bd.select_partitions(aligned, cols, ['test'])
    test_rows, _ = bd.select_partitions(aligned, cols, ['test'], allow_sealed=True)
    matrix = bd.load_layer_matrix(test_rows, activations/'layer_08')
    assert matrix[:, 0].tolist() == [numbers[q]+100 for q in test_rows['ep_question_id']]
    config = ls.LayerSelectionConfig(capstone_data_root=Path('.'), results_dir=str(results),
        activations_dir=str(activations), labels_csv=str(results/'labels.csv'),
        manifest_csv=str(results/'split_manifest.csv'), source_splits=['train', 'validation'], text_fields=[])
    dev, _, _ = ls.load_inputs(config)
    assert set(dev['partition']) == {'train', 'validation'}


def test_existing_label_package_cli_supports_distributed_layout(collected, tmp_path, monkeypatch):
    results, activations, _, _, _ = collected
    package = tmp_path/'behavior-package'
    monkeypatch.setattr('sys.argv', ['prepare_behavior_v01.py', '--full-results', str(results),
        '--layout', 'partitioned', '--label-policy', 'v0.2.0-candidate', '--output', str(package)])
    runpy.run_path(str(Path(__file__).resolve().parents[3]/'scripts/prepare_behavior_v01.py'), run_name='__main__')
    rows, manifest = load_labeled_rows(results, package, layout='partitioned')
    assert set(r['partition'] for r in rows) == {'train', 'validation', 'test'}
    counts = json.loads((package/'counts.json').read_text())
    assert counts['partition']['questions']['missing'] == 0
    assert 'test' in counts['by_storage_split']
    assert not (package/'missing_question_ids.json').exists()
    assert read_partitions(package/'partitions.csv', layout='partitioned')['musique:2hop__3_3']['partition'] == 'intervention'


def test_missing_test_folder_is_not_silently_ignored(collected):
    results, _, _, _, _ = collected
    (results/'test/interactions.jsonl').unlink()
    with pytest.raises(FileNotFoundError):
        read_run_interactions(results, layout='partitioned')


def test_wrong_partition_record_is_rejected(collected):
    results, _, _, _, _ = collected
    rows = read_run_interactions(results, layout='partitioned')
    frozen = read_partitions(results/'partitions.csv', layout='partitioned')
    rows[0]['source_split'] = 'validation' if rows[0]['source_split'] == 'test' else 'test'
    with pytest.raises(ValueError):
        build_partition_manifest(rows, frozen=frozen, layout='partitioned')


def test_export_cli_keeps_distributed_layout_usable(collected, tmp_path, monkeypatch):
    results, activations, numbers, _, _ = collected
    attach_fixture_labels(results)
    out_a, out_r = tmp_path/'export-a', tmp_path/'export-r'
    monkeypatch.setattr('sys.argv', ['export_partition_activations.py', '--package', str(results),
        '--source-results', str(results), '--source-activations', str(activations), '--run', 'full',
        '--layers', '8', '--layout', 'partitioned', '--frozen-partitions', str(results/'partitions.csv'),
        '--export-run-name', 'copy', '--export-activation-root', str(out_a), '--export-result-root', str(out_r)])
    runpy.run_path(str(Path(__file__).resolve().parents[3]/'scripts/export_partition_activations.py'), run_name='__main__')
    assert (out_a/'copy/layer_08/test.pt').is_file()
    assert not (out_a/'copy/layer_08/intervention.pt').exists()
    test_rows = read_jsonl(out_r/'copy/test/interactions.jsonl')
    matrix = torch.load(out_a/'copy/layer_08/test_attempt2.pt', weights_only=True)
    assert matrix[:, 0].tolist() == [numbers[r['question_id']]+100 for r in test_rows]


def test_unknown_exposure_is_reported_and_not_sampled_as_untouched(inputs, tmp_path):
    from mas_sae.data.partition_adapter import canonical_partitions
    from mas_sae.data.production import sample_partition_questions
    master, historical, _ = inputs
    imported = import_historical_partitions(master, historical)
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=42, existing=imported)
    fixed = canonical_partitions(frozen['questions'])
    unknown_test = fixed['musique:2hop__2_2']
    assert unknown_test['in_development_data'] is None
    assert unknown_test['exposure_history_known'] is False
    with pytest.raises(ValueError, match='eligible'):
        sample_partition_questions({unknown_test['question_id']: unknown_test}, partition='test', num_questions=1, seed=1)


def test_explicit_wrong_label_episode_is_rejected(collected):
    results, _, _, _, _ = collected
    attach_fixture_labels(results)
    labels = pd.read_csv(results/'labels.csv')
    labels.loc[0, 'episode_id'] = 'another-episode'
    labels.to_csv(results/'labels.csv', index=False)
    with pytest.raises(ValueError, match='label episode_id'):
        load_labeled_rows(results, results, layout='partitioned')


def test_intervention_cannot_be_loaded_for_sae_fitting(collected):
    _, activations, _, _, _ = collected
    with pytest.raises(FileNotFoundError):
        create_sae_dataloader(8, 'intervention', 0, location=activations/'layer_08')


def test_partition_selection_cli_uses_new_ids_and_exclusions(collected, tmp_path, monkeypatch):
    results, _, _, _, _ = collected
    output = tmp_path/'selected.json'
    args = ['select_partition_questions.py', '--package', str(results), '--partition', 'train',
        '--num-questions', '1', '--seed', '42', '--layout', 'partitioned',
        '--frozen-partitions', str(results/'partitions.csv'), '--output', str(output)]
    monkeypatch.setattr('sys.argv', args)
    script = str(Path(__file__).resolve().parents[3]/'scripts/select_partition_questions.py')
    runpy.run_path(script, run_name='__main__')
    ids = json.loads(output.read_text())
    assert len(ids) == 1 and ids[0].startswith('synthetic:')
    # A namespaced synthetic exclusion must not be lost by a MuSiQue-only regex.
    exclusions = tmp_path/'excluded.json'
    exclusions.write_text(json.dumps(ids))
    monkeypatch.setattr('sys.argv', args[:-1]+[str(tmp_path/'second.json'), '--exclude', str(exclusions)])
    with pytest.raises(ValueError, match='eligible'):
        runpy.run_path(script, run_name='__main__')


def test_later_historical_qc_flags_map_to_musique_only(collected, tmp_path, monkeypatch):
    results, _, _, _, _ = collected
    qc = tmp_path/'qc.json'
    qc.write_text(json.dumps({'samples': [{'question_id': '2hop__2_2'}, {'question_id': '2hop__3_3'}]}))
    package = tmp_path/'qc-package'
    monkeypatch.setattr('sys.argv', ['prepare_behavior_v01.py', '--full-results', str(results),
        '--layout', 'partitioned', '--label-policy', 'v0.2.0-candidate', '--first-qc-sample', str(qc), '--output', str(package)])
    runpy.run_path(str(Path(__file__).resolve().parents[3]/'scripts/prepare_behavior_v01.py'), run_name='__main__')
    manifest = read_manifest(package/'split_manifest.csv')
    assert manifest['musique:2hop__2_2']['in_development_data'] is True
    assert manifest['synthetic:2hop__1_1']['in_development_data'] is False
    fixed = read_partitions(package/'partitions.csv', layout='partitioned')
    assert fixed['musique:2hop__3_3']['in_qc100'] is True
    assert fixed['musique:2hop__3_3']['partition'] == 'intervention'
