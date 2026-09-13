"""Offline tests for the public validation, parsing, metric and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from deplot_chart_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    INSTRUCTION,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    RELATIVE_TOLERANCE,
    cell_accuracy,
    cells_match,
    evaluation_report,
    parse_table,
    validate_inputs,
)

LINEAR = "TITLE | Quarterly revenue <0x0A> Quarter | Revenue <0x0A> Q1 | 120 <0x0A> Q2 | 135 <0x0A> Q3 | 150"
EXPECTED = [["Quarter", "Revenue"], ["Q1", "120"], ["Q2", "135"], ["Q3", "150"]]


def _image(width: int = 800, height: int = 520) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(text: str = LINEAR, truncated: bool = False) -> dict:
    return {"text": text, "table": parse_table(text), "truncated": truncated}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["chart.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["instruction"] == INSTRUCTION
    assert manifest["inputs"] == [{"id": "chart.png", "mode": "RGB", "size": [800, 520]}]
    assert manifest["instruction"] == INSTRUCTION
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_request() -> None:
    manifest = validate_inputs(_image(), max_new_tokens=256)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["generation"]["max_new_tokens"] == 256


def test_validate_inputs_rejects_like_extract_table() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), max_new_tokens=0)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_parse_table_splits_title_rows_and_cells() -> None:
    table = parse_table(LINEAR)
    assert table["title"] == "Quarterly revenue"
    assert table["rows"] == EXPECTED and (table["n_rows"], table["n_columns"]) == (4, 2)
    untitled = parse_table("A | B <0x0A> 1 | 2 | 3 <0x0A>  <0x0A>")
    assert untitled["title"] is None and untitled["rows"] == [["A", "B"], ["1", "2", "3"]]
    assert untitled["n_columns"] == 3
    assert parse_table("")["rows"] == [] and parse_table("")["n_columns"] == 0


def test_cells_match_relaxed_numeric_and_text() -> None:
    assert cells_match("Revenue", " revenue ") is True
    assert cells_match("120", "118") is True  # within 5 %
    assert cells_match("120", "110") is False
    assert cells_match("$1,099.20", "1099.2") is True
    assert cells_match("12%", "12") is True
    assert (
        cells_match("0", "0") is True
        and cells_match("0.04", "0") is True
        and cells_match("0.06", "0") is False
    )
    assert cells_match("Q1", "120") is False
    assert 0 < RELATIVE_TOLERANCE < 1


def test_cell_accuracy_is_position_wise() -> None:
    accuracy = cell_accuracy(parse_table(LINEAR)["rows"], EXPECTED)
    assert accuracy == {
        "matched": 8,
        "total": 8,
        "value": 1.0,
        "predicted_shape": [4, 2],
        "expected_shape": [4, 2],
        "relative_tolerance": RELATIVE_TOLERANCE,
    }
    short = cell_accuracy([["Quarter", "Revenue"], ["Q1", "121"]], EXPECTED)
    assert (short["matched"], short["total"], short["value"]) == (4, 8, 0.5)
    with pytest.raises(ValueError, match="at least one cell"):
        cell_accuracy([], [])


def test_evaluation_report_not_measurable_without_expected_rows() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["predicted_shape"] == [4, 2]
    assert "relative mapping similarity" in report["needs"]
    assert report["baselines"] == []
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no score" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_expected_rows_and_title() -> None:
    report = evaluation_report(
        _result(), EXPECTED, expected_title="quarterly revenue", sample_kind="synthetic"
    )
    assert report["verdict"] == "sample-sanity" and report["sample_kind"] == "synthetic"
    by_id = {m["id"]: m for m in report["metrics"]}
    assert by_id["cell_accuracy"]["value"] == 1.0 and by_id["cell_accuracy"]["total"] == 8
    assert by_id["title_match"]["value"] == 1.0
    wrong_title = evaluation_report(_result(), EXPECTED, expected_title="Annual revenue")
    assert {m["id"]: m for m in wrong_title["metrics"]}["title_match"]["value"] == 0.0
    no_title = evaluation_report({"text": "A | B <0x0A> 1 | 2"}, [["A", "B"], ["1", "2"]], expected_title="X")
    assert {m["id"]: m for m in no_title["metrics"]}["title_match"]["predicted"] is None


def test_evaluation_report_carries_truncation_flag() -> None:
    assert evaluation_report(_result(truncated=True))["truncated"] is True
