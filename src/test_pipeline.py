"""Tests for the focused writing-to-dictation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.f_text_dictation_visualisations import (
    _collect_text_rows,
    generate_all_dictation_visualisations,
)
from src.e_orchestrator import run_dictation_pipeline


def test_collect_text_rows_recomputes_missing_pos_and_rhythm_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    text_name = "henry_james_the_american_1877"
    text_dir = tmp_path / "data" / "analytics" / "window_metrics" / "james" / "james" / text_name
    text_dir.mkdir(parents=True, exist_ok=True)

    (text_dir / "window_metrics.syntax.json").write_text(
        json.dumps(
            {
                "sentences": [
                    {
                        "token_count": 12,
                        "clause_counts": {"main": 1, "subordinate": 1, "coordinate": 0},
                        "clause_ratios": {"coordination_ratio": 0.0, "subordination_ratio": 1.0},
                        "avg_dependents_per_head": {
                            "main_clause": 1.2,
                            "subordinate_clause": 0.8,
                            "coordinate_clause": 0.0,
                        },
                        "max_depth": 3,
                        "avg_mean_dependency_distance": 1.4,
                        "normalized_structural_entropy": 0.41,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (text_dir / "window_metrics.lexico_semantics.json").write_text(
        json.dumps({"sentences": [{"lexical_density": 0.57, "normalized_lexical_entropy": 0.61}]}),
        encoding="utf-8",
    )
    (text_dir / "window_metrics.discourse.json").write_text(
        json.dumps(
            {
                "sentences": [
                    {
                        "explicit_connectives_per_token": 0.05,
                        "connective_counts_per_token": {"Expansion": 0.02, "Contingency": 0.01},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (text_dir / "window_metrics.log_prob.json").write_text(
        json.dumps(
            {
                "sentences": [
                    {
                        "sentence_surprisal_metrics": {
                            "num_tokens": 12,
                            "mean_surprisal": 3.1,
                            "surprisal_variance": 0.4,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.f_text_dictation_visualisations._recompute_syntax_sentences",
        lambda author, text_name: [
            {
                "token_count": 12,
                "clause_counts": {"main": 1, "subordinate": 1, "coordinate": 0},
                "clause_ratios": {"coordination_ratio": 0.0, "subordination_ratio": 1.0},
                "avg_dependents_per_head": {
                    "main_clause": 1.2,
                    "subordinate_clause": 0.8,
                    "coordinate_clause": 0.0,
                },
                "max_depth": 3,
                "avg_mean_dependency_distance": 1.4,
                "normalized_structural_entropy": 0.41,
                "normalized_pos_ngram_entropy": 0.83,
                "sentence_length_approx_entropy": 0.12,
            }
        ],
    )

    rows = _collect_text_rows(tmp_path / "data" / "analytics" / "window_metrics")

    assert rows
    assert rows[0]["normalized_pos_ngram_entropy"] == pytest.approx(0.83)
    assert rows[0]["sentence_length_approx_entropy"] == pytest.approx(0.12)


def test_generate_dictation_visualisations_outputs_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "author": "james",
            "group": "part_1_physical_cognitive",
            "phase": "early",
            "text_name": "henry_james_the_american_1877",
            "avg_sentence_length": 22.0,
            "avg_subordinate_per_sentence": 1.4,
            "lexical_density": 0.56,
            "mean_dependency_distance": 2.3,
            "parataxis_count": 8,
            "hypotaxis_count": 14,
            "parataxis_to_hypotaxis_ratio": 9 / 15,
            "normalized_structural_entropy_series": [0.42, 0.46, 0.5, 0.44],
            "breath_unit_length_series": [14, 13, 12, 11, 12, 10],
        },
        {
            "author": "james",
            "group": "part_1_physical_cognitive",
            "phase": "late",
            "text_name": "henry_james_the_golden_bowl_1904",
            "avg_sentence_length": 18.0,
            "avg_subordinate_per_sentence": 0.9,
            "lexical_density": 0.63,
            "mean_dependency_distance": 1.8,
            "parataxis_count": 16,
            "hypotaxis_count": 8,
            "parataxis_to_hypotaxis_ratio": 17 / 9,
            "normalized_structural_entropy_series": [0.31, 0.36, 0.34, 0.29],
            "breath_unit_length_series": [11, 10, 9, 9, 8, 8],
        },
    ]

    monkeypatch.setattr(
        "src.f_text_dictation_visualisations._collect_text_rows",
        lambda *args, **kwargs: rows,
    )

    outputs = generate_all_dictation_visualisations(output_root=tmp_path, use_existing=False)

    assert outputs["part_1_physical_cognitive"]["mean_dependency_distance_slopegraph"] is not None
    assert outputs["part_1_physical_cognitive"]["parataxis_hypotaxis_divergence"] is not None
    assert outputs["part_1_physical_cognitive"]["distribution_divergence_bars"] is not None


def test_run_dictation_pipeline_executes_only_relevant_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    monkeypatch.setattr("src.e_orchestrator.run_preprocessing", lambda **kwargs: calls.append("pre"))
    monkeypatch.setattr("src.e_orchestrator.run_corpus_metrics", lambda **kwargs: calls.append("corpus"))
    monkeypatch.setattr("src.e_orchestrator.run_windowed_metrics", lambda **kwargs: calls.append("window"))
    monkeypatch.setattr(
        "src.e_orchestrator.generate_all_dictation_visualisations",
        lambda **kwargs: calls.append("viz") or {"ok": True},
    )

    outputs = run_dictation_pipeline(use_existing=False, output_root=tmp_path)

    assert calls == ["pre", "corpus", "window", "viz"]
    assert outputs == {"ok": True}
