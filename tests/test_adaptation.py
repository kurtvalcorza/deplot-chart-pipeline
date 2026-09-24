"""Offline checks of the OTSL-to-DePlot target converter, the chart/table dataset contract, the metrics and
baselines, the pinned-shard reader (refusal while unpinned, digest checks when pinned) and the adaptation
surface — no model weights, no network."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image

from deplot_chart_pipeline import (
    MAX_RECORDS,
    MAX_TARGET_CHARS,
    MIN_RECORDS,
    DePlotPipeline,
    build_sample_dataset,
    chart_metrics,
    check_split_disjoint,
    corpus_pinned,
    dataset_digest,
    empty_baseline,
    fetch_corpus,
    format_table,
    header_only_baseline,
    load_byod_dataset,
    medoid_baseline,
    otsl_to_rows,
    parse_otsl,
    parse_table,
    read_corpus,
    rnss,
    score_table,
    split_dataset,
    stratified_quotas,
    target_from_otsl,
    validate_dataset,
    write_dataset_jsonl,
)
from deplot_chart_pipeline import samples as samples_module

LOC = "<loc_0><loc_0><loc_500><loc_500>"
BAR = f"<chart>{LOC}<bar_chart><fcel>North<fcel>South<fcel>East<nl><fcel>12<fcel>7.5<fcel>30%<nl></chart>"
PIE = f"{LOC}<pie_chart><fcel>A<nl><fcel>100%<nl></chart>"
STACKED = (
    f"<chart>{LOC}<stacked_bar_chart><ecel><fcel>Q1<fcel>Q2<nl>"
    "<fcel>Sales<fcel>10<fcel>20<nl><fcel>Costs<fcel>4<fcel>5<nl></chart>"
)
LINE = f"{LOC}<line_chart><ecel><fcel>Debt<nl><fcel>2019<fcel>1.5<nl><fcel>2020<fcel>2.5<nl></chart>"


def _png(colour, size=(64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg(seed: int, size=(96, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (seed * 7 % 255, seed * 13 % 255, 200 - seed % 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def records(tmp_path):
    out = []
    for i in range(12):
        path = tmp_path / f"c{i}.png"
        path.write_bytes(_png((i * 20, 40, 200 - i * 10)))
        out.append(
            {
                "id": f"chart-{i:02d}",
                "image_id": f"c{i}",
                "image": str(path),
                "target_text": format_table([["North", str(10 + i)], ["South", str(20 + i)]]),
                "chart_type": "bar" if i % 2 else "pie",
            }
        )
    return out


# ---- converter ----


def test_bar_and_pie_tables_are_transposed_to_one_row_per_category():
    assert otsl_to_rows(BAR) == [["North", "12"], ["South", "7.5"], ["East", "30%"]]
    assert target_from_otsl(BAR) == "TITLE |  <0x0A> North | 12 <0x0A> South | 7.5 <0x0A> East | 30%"
    assert otsl_to_rows(PIE) == [["A", "100%"]]


def test_stacked_bar_is_transposed_and_keeps_its_empty_corner_header():
    assert otsl_to_rows(STACKED) == [["", "Sales", "Costs"], ["Q1", "10", "4"], ["Q2", "20", "5"]]
    assert target_from_otsl(STACKED).startswith("TITLE |  <0x0A>  | Sales | Costs <0x0A> Q1 | 10 | 4")


def test_line_tables_keep_their_orientation():
    assert otsl_to_rows(LINE) == [["", "Debt"], ["2019", "1.5"], ["2020", "2.5"]]
    table = parse_table(target_from_otsl(LINE))
    assert table["title"] == "" and table["rows"] == [["", "Debt"], ["2019", "1.5"], ["2020", "2.5"]]


def test_target_round_trips_through_parse_table():
    for otsl in (BAR, PIE, STACKED, LINE):
        rows = otsl_to_rows(otsl)
        parsed = parse_table(target_from_otsl(otsl))["rows"]
        assert parsed == [r for r in rows if any(r)]


@pytest.mark.parametrize(
    ("otsl", "message"),
    [
        (f"{LOC}<fcel>a<nl>", "no <"),
        (f"{LOC}<bar_chart><fcel>a<lcel><nl><fcel>1<fcel>2<nl>", "other than"),
        (f"{LOC}<bar_chart><fcel>a<fcel>b<nl><fcel>1<nl>", "ragged"),
        (f"{LOC}<bar_chart></chart>", "no rows"),
        (f"{LOC}<bar_chart><fcel>a|b<nl><fcel>1<nl>", "separator"),
    ],
)
def test_parse_otsl_refuses_what_the_converter_cannot_represent(otsl, message):
    with pytest.raises(ValueError, match=message):
        parse_otsl(otsl)


def test_stratified_quotas_use_the_largest_remainder():
    counts = {"bar": 6491, "pie": 6665, "stacked_bar": 1333, "line": 79}
    assert stratified_quotas(counts, 160) == {"bar": 71, "pie": 73, "stacked_bar": 15, "line": 1}
    assert sum(stratified_quotas(counts, 80).values()) == 80
    with pytest.raises(ValueError):
        stratified_quotas({}, 10)


# ---- metrics and baselines ----


def test_score_table_cell_accuracy_rnss_and_exact_match():
    target = format_table([["North", "10"], ["South", "20"]])
    exact = score_table(target, target)
    assert exact == {**exact, "cell_accuracy": 1.0, "rnss": 1.0, "exact_table_match": True}
    near = score_table(format_table([["north", "10.4"], ["South", "30"]]), target)
    assert near["cell_accuracy"] == 0.75 and not near["exact_table_match"]
    assert near["rnss"] == pytest.approx(1 - (0.04 + 0.5) / 2)
    assert score_table("", target)["cell_accuracy"] == 0.0 and score_table("", target)["rnss"] == 0.0


def test_rnss_matches_numbers_regardless_of_order_and_charges_unmatched():
    assert rnss([["b", "20"], ["a", "10"]], [["a", "10"], ["b", "20"]]) == 1.0
    assert rnss([["a", "10"]], [["a", "10"], ["b", "20"]]) == pytest.approx(0.5)
    assert rnss([["x"]], [["y"]]) == 1.0
    assert rnss([["0"]], [["0"], ["5"]]) == pytest.approx(0.5)


def test_chart_metrics_reports_means_and_per_type(records):
    checked = validate_dataset(records)["records"]
    metrics = chart_metrics([r["target_text"] for r in checked], checked)
    assert metrics["n"] == 12 and metrics["cell_accuracy"] == 1.0 and metrics["exact_table_match"] == 1.0
    assert set(metrics["by_chart_type"]) == {"bar", "pie"} and metrics["by_chart_type"]["bar"]["n"] == 6
    assert set(metrics["definitions"]) >= {"cell_accuracy", "rnss", "exact_table_match"}
    with pytest.raises(ValueError):
        chart_metrics([""], checked)


def test_baselines_never_read_the_test_targets(records):
    checked = validate_dataset(records)["records"]
    train, test = checked[:8], checked[8:]
    assert empty_baseline(test)["cell_accuracy"] == 0.0
    header = header_only_baseline(train, test)
    assert header["baseline"] == "header-only" and header["cell_accuracy"] == 0.0  # no header rows here
    medoid = medoid_baseline(train, test)
    assert medoid["baseline"] == "medoid" and set(medoid["tables"]) == {"bar", "pie"}
    assert 0.0 < medoid["cell_accuracy"] < 1.0  # the category labels match, the values do not
    stacked = [{**r, "chart_type": "stacked_bar", "target_text": target_from_otsl(STACKED)} for r in train]
    assert header_only_baseline(stacked, stacked[:2])["headers"]["stacked_bar"].endswith(" | Sales | Costs")


# ---- dataset contract ----


def test_validate_dataset_reports_structure_and_digest(records):
    manifest = validate_dataset(records)
    assert manifest["n_records"] == 12 and manifest["unique_images"] == 12
    assert manifest["chart_types"] == {"pie": 6, "bar": 6}
    assert manifest["table_rows"] == {"min": 2, "max": 2}
    assert manifest["digest"] == dataset_digest(manifest["records"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: {**r, "id": "bad id!"}, "id must match"),
        (lambda r: {**r, "image": "/does/not/exist.png"}, "image file not found"),
        (lambda r: {**r, "target_text": "   "}, "non-empty string"),
        (lambda r: {**r, "target_text": "x" * (MAX_TARGET_CHARS + 1)}, "MAX_TARGET_CHARS"),
        (lambda r: {**r, "target_text": "TITLE | only a title"}, "at least one table row"),
        (lambda r: {k: v for k, v in r.items() if k != "target_text"}, "missing 'target_text'"),
    ],
)
def test_validate_dataset_refuses_contract_violations(records, mutate, message):
    with pytest.raises(ValueError, match=message):
        validate_dataset([mutate(records[0]), *records[1:]])


def test_validate_dataset_refuses_duplicates_and_size_bounds(records):
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records[:8]])
    with pytest.raises(ValueError, match=f"{MIN_RECORDS}..{MAX_RECORDS}"):
        validate_dataset(records[:3])


def test_split_dataset_keeps_each_image_in_one_split(records):
    shared = [{**r, "image_id": f"g{i // 2}"} for i, r in enumerate(records)]
    splits = split_dataset(shared, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert sum(check_split_disjoint(splits).values()) == 12
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": shared[:1], "test": shared[1:2]})


def test_byod_jsonl_round_trip(records, tmp_path):
    path = write_dataset_jsonl(records, tmp_path / "out" / "records.jsonl")
    again = load_byod_dataset(path)
    assert [r["id"] for r in again] == [r["id"] for r in records]
    assert validate_dataset(again)["digest"] == validate_dataset(records)["digest"]


# ---- pinned shard ----


def _shard(tmp_path, per_type=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    per_type = per_type or {"bar": 40, "pie": 40, "stacked_bar": 12, "line": 8}
    templates = {"bar": BAR, "pie": PIE, "stacked_bar": STACKED, "line": LINE}
    rows, k = [], 0
    for chart_type, count in per_type.items():
        for _ in range(count):
            otsl = templates[chart_type].replace("12", str(100 + k))
            texts = [{"user": "Convert chart.", "assistant": otsl, "source": "test"}]
            rows.append({"images": [{"bytes": _jpeg(k), "path": None}], "texts": texts})
            k += 1
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def test_fetch_corpus_refuses_while_no_pin_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", None)
    assert not corpus_pinned()
    with pytest.raises(RuntimeError, match="no SHA-256 pin is recorded"):
        fetch_corpus(cache_dir=tmp_path, downloader=lambda cache: pytest.fail("must not download"))


def test_fetch_corpus_checks_size_and_digest(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    data = shard.read_bytes()
    monkeypatch.setitem(samples_module.CORPUS_FILE, "bytes", len(data))
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", hashlib.sha256(data).hexdigest())
    assert corpus_pinned()
    assert fetch_corpus(cache_dir=tmp_path / "cache", downloader=lambda cache: shard) == shard
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="refusing to read it"):
        fetch_corpus(cache_dir=tmp_path / "cache", downloader=lambda cache: shard)


def test_pinned_constants_match_the_decided_shard():
    assert samples_module.CORPUS_REVISION == "b913ef98a3d45f6136465963ddc71a7a6b0e1728"
    assert samples_module.CORPUS_FILE["bytes"] == 515_825_444 and samples_module.CORPUS_FILE["rows"] == 14_568
    assert (
        samples_module.CORPUS_FILE["sha256"]
        == "1d6cf578fdd953fa0dd7c1d169926eb3418daa2770cc3a301b1a64bec0b209e7"
    )
    assert samples_module.SAMPLE_CHARTS == {"train": 360, "validation": 80, "test": 160}


def test_read_corpus_and_build_a_stratified_image_disjoint_sample(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "rows", None)
    rows = read_corpus(shard)
    assert len(rows) == 100 and rows[0]["chart_type"] == "bar" and rows[-1]["chart_type"] == "line"
    sizes = {"train": 50, "validation": 10, "test": 20}
    splits = build_sample_dataset(rows, shard, seed=3, sizes=sizes, image_dir=tmp_path / "images")
    assert check_split_disjoint(splits) == {"train": 50, "validation": 10, "test": 20}
    assert {r["chart_type"] for r in splits["test"]} == {"bar", "pie", "stacked_bar", "line"}
    assert all(r["target_text"] == target_from_otsl(rows[r["row"]]["otsl"]) for r in splits["train"])
    again = build_sample_dataset(rows, shard, seed=3, sizes=sizes, image_dir=tmp_path / "images")
    assert [r["row"] for r in again["test"]] == [r["row"] for r in splits["test"]]
    for part in splits.values():
        validate_dataset(part)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "rows", 99)
    with pytest.raises(ValueError, match="pinned 99"):
        read_corpus(shard)
    with pytest.raises(ValueError, match="do not fill"):
        build_sample_dataset(
            rows, shard, sizes={"train": 200, "validation": 1, "test": 1}, image_dir=tmp_path / "i2"
        )


def test_pinned_sample_validates_when_the_shard_is_cached(tmp_path):
    local = samples_module.DEFAULT_CACHE_DIR / samples_module.CORPUS_FILE["path"]
    if not corpus_pinned() or not local.is_file():
        pytest.skip("pinned SynthChartNet shard not cached locally")
    shard = fetch_corpus()
    splits = build_sample_dataset(read_corpus(shard), shard, image_dir=tmp_path / "images")
    for name, part in splits.items():
        assert len(part) == samples_module.SAMPLE_CHARTS[name]
        validate_dataset(part)


# ---- adaptation surface without a model ----


def _runner(image, max_new_tokens):
    return {"text": format_table([["North", "10"], ["South", "0"]]), "new_tokens": 9}


def test_evaluate_with_an_injected_runner_scores_tables(records):
    pipe = DePlotPipeline(_runner)
    metrics = pipe.evaluate(records)
    assert metrics["n"] == 12 and 0.0 < metrics["cell_accuracy"] < 1.0
    assert metrics["cell_accuracy"] == pytest.approx(
        0.5 + 0.25 / 12, abs=1e-4
    )  # labels right, one value right
    assert metrics["truncated_rate"] == 0.0 and metrics["verdict"] == "measured-small-sample"
    assert metrics["adapted"] is False
    predictions = pipe.predict(records[:2])
    assert predictions[0]["id"] == "chart-00" and predictions[0]["chart_type"] == "pie"


def test_adaptation_needs_a_loaded_model(records, tmp_path):
    pipe = DePlotPipeline(_runner)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    with pytest.raises(ValueError, match="1..12"):
        pipe._trainable_names(13)


def test_load_artifact_refuses_a_foreign_manifest(tmp_path):
    pipe = DePlotPipeline(_runner)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
