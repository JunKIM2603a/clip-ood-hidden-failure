"""CPU/synthetic tests only. These are not evidence for H_geo."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/run_hgeo_text_risk_pilot.py"
SPEC = importlib.util.spec_from_file_location("hgeo_pilot_test_target", PATH)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def make_inputs(root: Path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", root)
    monkeypatch.delenv("CLIP_OOD_DATA_ROOT", raising=False)
    names = root / "configs/subgroups/mappings/sun_mos50_hierarchy.csv"
    names.parent.mkdir(parents=True)
    pd.DataFrame({"leaf_concept": ["river", "ocean"]}).to_csv(names, index=False)
    semantic = root / "data/semantic_labels/sun_image_semantic_labels.csv"
    semantic.parent.mkdir(parents=True)
    pd.DataFrame({"relative_path": ["a", "b"], "leaf_concept": ["river", "ocean"]}).to_csv(semantic, index=False)
    for method in pilot.METHODS:
        base = root / "results/raw/reproduction/ViT-B-32" / method
        base.mkdir(parents=True)
        pd.DataFrame({"relative_path": ["i", "j"], "score": [.1, .9]}).to_csv(base / "imagenet.csv", index=False)
        pd.DataFrame({"relative_path": ["b", "a"], "score": [.2, .8]}).to_csv(base / "sun.csv", index=False)
    return semantic


def test_rank_matches_scipy_with_ties():
    x, y = np.array([1, 1, 3, 4, 8]), np.array([4, 2, 2, 9, 1])
    assert float(pilot.rank_corr(x, y)) == pytest.approx(spearmanr(x, y).statistic)
    assert np.allclose(pilot.rank_corr(np.stack([x, x]), np.stack([y, y])), spearmanr(x, y).statistic)


def test_constant_rank_is_undefined():
    assert np.isnan(pilot.rank_corr(np.ones(10), np.arange(10)))


def test_bad_rank_inputs_raise():
    with pytest.raises(ValueError):
        pilot.rank_corr(np.array([1, np.nan]), np.array([1, 2]))
    with pytest.raises(ValueError):
        pilot.rank_corr(np.arange(2), np.arange(3))


def test_cosine_geometry_matches_definition():
    concepts, positive = np.eye(2), np.array([[1., 0.], [0., 1.], [-1., 0.]])
    result = pilot.cosine_geometry(concepts, positive)
    sim = concepts @ positive.T
    np.testing.assert_allclose(result["peak"], sim.max(1) - sim.mean(1))
    np.testing.assert_allclose(result["id_q95"], np.quantile(sim, .95, axis=1))


def test_unnormalized_features_rejected():
    with pytest.raises(ValueError, match="normalized"):
        pilot.cosine_geometry(np.array([[2., 0.]]), np.eye(2))


def test_error_join_uses_path_and_includes_threshold_ties():
    scores = pd.DataFrame({"relative_path": ["c", "b", "a"], "score": [.4, .5, .8]})
    semantic = pd.DataFrame({"relative_path": ["a", "b", "c"], "leaf_concept": ["x", "y", "y"]})
    result = pilot.concept_errors(scores, semantic, .5).set_index("leaf_concept")
    assert result.loc["x", "fpr95"] == 1
    assert result.loc["y", "fpr95"] == .5
    assert result.loc["y", "n"] == 2
    assert not result.eligible.any()


def test_unmapped_sample_rejected():
    scores = pd.DataFrame({"relative_path": ["a"], "score": [.5]})
    semantic = pd.DataFrame({"relative_path": ["b"], "leaf_concept": ["x"]})
    with pytest.raises(ValueError, match="Missing leaf"):
        pilot.concept_errors(scores, semantic, .5)


def test_duplicate_input_rejected(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame({"relative_path": ["a", "a"], "score": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        pilot.read_frame(path, {"relative_path", "score"}, "relative_path")


def make_rank_table():
    return pd.DataFrame({"leaf_concept": [f"c{i:02d}" for i in range(20)], "eligible": True,
                         "fpr95": np.arange(20) / 20, "max_id_cosine": np.arange(20),
                         "text_proxy_score": np.arange(20)})


def test_paired_delta_is_zero_for_identical_predictors():
    result = pilot.summarize_ranks(make_rank_table(), ["text_proxy_score", "max_id_cosine"], 100)
    assert np.allclose(result.rho, 1)
    assert np.allclose(result.delta_ci_low, 0)
    assert np.allclose(result.delta_ci_high, 0)
    assert result.iloc[0].status == "PROMISING_DESCRIPTIVE_SIGNAL"


def test_no_failures_is_undefined_not_false_evidence():
    table = make_rank_table()
    table["fpr95"] = 0
    result = pilot.summarize_ranks(table, ["text_proxy_score"], 100)
    assert result.iloc[0].status == "UNDEFINED_OR_UNSTABLE"
    assert np.isnan(result.iloc[0].rho)


def test_bootstrap_reproducible_and_minimum_sample_checked():
    table = make_rank_table()
    table["text_proxy_score"] = np.random.default_rng(123).normal(size=len(table))
    a = pilot.summarize_ranks(table, ["text_proxy_score"], 100)
    b = pilot.summarize_ranks(table, ["text_proxy_score"], 100)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="eligible concepts"):
        pilot.summarize_ranks(table.iloc[:3], ["text_proxy_score"], 100)


def test_input_collection_and_provenance(tmp_path, monkeypatch):
    make_inputs(tmp_path, monkeypatch)
    frames, leaves, provenance = pilot.collect_inputs("ViT-B/32")
    assert leaves == ["ocean", "river"]
    assert len(frames["mcm_sun"]) == 2
    assert len(provenance) == 6
    assert len(provenance["concept_names"]["sha256"]) == 64


def test_check_inputs_never_calls_gpu(tmp_path, monkeypatch, capsys):
    make_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["pilot", "--check-inputs"])
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU path must not run during input checks")
    monkeypatch.setattr(pilot, "compute_text_risks", forbidden)
    pilot.main()
    assert "inputs OK" in capsys.readouterr().out
    assert not (tmp_path / "results/hgeo_text_risk").exists()


def test_method_sample_mismatch_rejected(tmp_path, monkeypatch):
    make_inputs(tmp_path, monkeypatch)
    path = tmp_path / "results/raw/reproduction/ViT-B-32/neglabel/sun.csv"
    pd.DataFrame({"relative_path": ["a", "z"], "score": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="different sun sample sets"):
        pilot.collect_inputs("ViT-B/32")


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    make_inputs(tmp_path, monkeypatch)
    output = tmp_path / "results/hgeo_text_risk/ViT-B-32/sun_pilot_v1"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    monkeypatch.setattr("sys.argv", ["pilot"])
    with pytest.raises(FileExistsError):
        pilot.main()
    assert sentinel.read_text() == "original"


def test_json_undefined_statistics_are_null():
    value = pilot.json_safe({"rho": np.nan, "n": np.int64(2), "nested": [np.inf]})
    assert json.loads(json.dumps(value, allow_nan=False)) == {"rho": None, "n": 2, "nested": [None]}
