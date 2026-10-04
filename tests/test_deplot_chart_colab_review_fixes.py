"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/deplot_chart_colab.ipynb`
(review PR #6: DPC-M1..M3, DPC-m1..m4).

The notebook's own cells are executed from the committed JSON in a namespace of the package's public API and
inert stand-ins (an injected table runner, a fake `google.colab`, tiny PIL drawings, a `display` stub). The
re-run regression (DPC-M2) runs the real `adapt` / `save_artifact` / `load_artifact` path on a tiny random
Pix2Struct built from the committed config, like `tests/test_adaptation_model.py`; it is skipped without torch.
Nothing here loads the pinned checkpoint.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before any NumPy linear algebra in this process.
with contextlib.suppress(ImportError):
    import torch

from PIL import Image, ImageDraw  # noqa: E402

import deplot_chart_pipeline as dcp  # noqa: E402
from deplot_chart_pipeline import (  # noqa: E402
    DEFAULT_WEIGHTS_DIR,
    DePlotPipeline,
    byod_record_limits,
    format_table,
    split_dataset,
)
from deplot_chart_pipeline import samples as samples_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "deplot_chart_colab.ipynb"
needs_torch = pytest.mark.skipif("torch" not in sys.modules, reason="torch is not installed")


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    """Source of the first code cell after the markdown cell containing `heading`."""
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _chart(path: Path, k: int) -> Path:
    """A small, distinct bar-chart drawing (distinct bytes, so every record is its own chart image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), (255 - k % 200, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 40 - (k % 30), 20, 40], fill=(60, 110, 200))
    draw.rectangle([30, 40 - (k * 7 % 30), 42, 40], fill=(200, 60, 60))
    image.save(path, format="PNG")
    return path


def _records(root: Path, n: int, *, prefix: str = "r", folder: str = "") -> list[dict]:
    out = []
    for i in range(n):
        rel = f"{folder}{prefix}{i:03d}.png"
        _chart(root / rel, i)
        out.append({"id": f"{prefix}{i:03d}", "image": rel, "target_text": format_table([["North", str(10 + i)], ["South", str(20 + i)]]), "chart_type": "bar" if i % 2 else "pie"})
    return out


def _zip(tmp_path: Path, name: str, n: int, *, prefix: str = "r", folder: str = "", records_name: str = "records.jsonl") -> bytes:
    src = tmp_path / f"src_{name}"
    records = _records(src, n, prefix=prefix, folder=folder)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if records_name:
            text = "\n".join(json.dumps(r) for r in records) if records_name.endswith(".jsonl") else json.dumps(records)
            archive.writestr(records_name, text)
        for r in records:
            archive.write(src / r["image"], r["image"])
    return buffer.getvalue()


def _fake_colab(monkeypatch, uploads: list[dict]):
    """A fake `google.colab.files.upload` returning the queued uploads in order."""
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` raises ImportError


def _section4(monkeypatch, tmp_path, *, byod_path: str = "", use_byod: bool = True) -> dict:
    """Execute Section 4 verbatim (form literals substituted) in a namespace of the package API."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. SynthChartNet charts, tables and split")
    source = source.replace("USE_BYOD = False", f"USE_BYOD = {use_byod}", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    shown: list = []
    ns: dict = {name: getattr(dcp, name) for name in dcp.__all__}
    ns.update({"os": __import__("os"), "Path": Path, "display": shown.append, "_shown": shown})
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


# --- DPC-m2: the BYOD bounds are stated, checked before any model runs, and name the split ----------


def test_byod_record_limits_are_twelve_to_2502():
    assert byod_record_limits() == (12, 2502)


def test_split_refusals_name_the_split_and_the_real_bounds(tmp_path):
    records = [{**r, "image": str(tmp_path / r["image"])} for r in _records(tmp_path, 14)]
    for n in range(8, 12):
        with pytest.raises(ValueError, match=r"the (train|validation) split has \d+ records \(at least \d+ are required\).*needs 12\.\.2502 records"):
            split_dataset(records[:n], seed=42)
    splits = split_dataset(records[:12], seed=42)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [8, 2, 2]


def test_split_refuses_a_scored_split_over_the_evaluation_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(samples_module, "MAX_EVAL_RECORDS", 5)
    records = [{**r, "image": str(tmp_path / r["image"])} for r in _records(tmp_path, 40)]
    with pytest.raises(ValueError, match=r"the validation split has 6 records \(at most 5 are scored\).*Use fewer records"):
        split_dataset(records, seed=42)


def test_twelve_records_pass_section4_and_eleven_are_refused_there(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"small.zip": _zip(tmp_path, "small", 11)}, {"ok.zip": _zip(tmp_path, "ok", 12)}])
    with pytest.raises(ValueError, match="the train split has 7 records"):
        _section4(monkeypatch, tmp_path)
    ns = _section4(monkeypatch, tmp_path)
    assert ns["disjoint"]["test"] == 2 and ns["disjoint"]["validation"] == 2 and ns["disjoint"]["train"] == 8
    assert ns["data_source"] == "BYOD (ok.zip)" and ns["byod"]["records"] == 12 and ns["byod"]["record_limits"] == [12, 2502]
    assert len(ns["_shown"]) == 1 and isinstance(ns["_shown"][0], Image.Image)  # DPC-m4: the example chart is displayed


def test_an_oversized_byod_set_is_refused_in_section4(monkeypatch, tmp_path):
    monkeypatch.setattr(samples_module, "MAX_EVAL_RECORDS", 5)  # stands in for 500, so 40 records play 2,600
    _fake_colab(monkeypatch, [{"big.zip": _zip(tmp_path, "big", 40)}])
    with pytest.raises(ValueError, match="at most 5 are scored"):
        _section4(monkeypatch, tmp_path)


def test_second_upload_replaces_the_first(monkeypatch, tmp_path):
    first = _zip(tmp_path, "a", 14, prefix="a", records_name="records.jsonl")
    second = _zip(tmp_path, "b", 13, prefix="b", records_name="records.json")
    _fake_colab(monkeypatch, [{"a.zip": first}, {"b.zip": second}])
    _section4(monkeypatch, tmp_path)
    ns = _section4(monkeypatch, tmp_path)
    ids = {r["id"] for part in (ns["train_records"], ns["val_records"], ns["test_records"]) for r in part}
    assert ns["data_source"] == "BYOD (b.zip)" and len(ids) == 13 and all(i.startswith("b") for i in ids)
    assert not (tmp_path / "work" / "byod" / "records.jsonl").exists()


def test_subfolder_image_paths_are_kept(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"charts.zip": _zip(tmp_path, "p", 12, folder="charts/")}])
    ns = _section4(monkeypatch, tmp_path)
    assert all("charts" in Path(r["image"]).parts for r in ns["train_records"])


def test_cancelled_upload_missing_records_and_no_colab_are_actionable(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}, {"no_records.zip": _zip(tmp_path, "n", 12, records_name="")}])
    with pytest.raises(RuntimeError, match=r"Upload exactly one .zip file \(received 0\)"):
        _section4(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="No records.jsonl or records.json found in no_records.zip"):
        _section4(monkeypatch, tmp_path)
    _no_colab(monkeypatch)
    with pytest.raises(RuntimeError, match="set BYOD_PATH to its path"):
        _section4(monkeypatch, tmp_path)


def test_byod_path_reads_a_zip_or_a_folder_without_colab(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    archive = tmp_path / "mine.zip"
    archive.write_bytes(_zip(tmp_path, "m", 12))
    ns = _section4(monkeypatch, tmp_path, byod_path=str(archive))
    assert ns["data_source"] == "BYOD (mine.zip)" and len(ns["byod"]["zip_sha256"]) == 64
    folder = tmp_path / "folder"
    records = _records(folder, 12)
    (folder / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(folder))
    assert ns["data_source"] == "BYOD (folder)" and ns["byod"]["zip_sha256"] is None
    with pytest.raises(FileNotFoundError, match="does not exist in this runtime"):
        _section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.zip"))


def test_zip_limits_and_unsafe_members_are_refused_before_writing(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"ok.zip": _zip(tmp_path, "ok", 12)}])
    ns = _section4(monkeypatch, tmp_path)
    extract = ns["extract_zip"]
    payload = _zip(tmp_path, "big", 3)
    root = tmp_path / "dest"
    root.mkdir()
    ns["MAX_ZIP_BYTES"] = 10
    with pytest.raises(ValueError, match="the limits are"):
        extract(payload, root, "big.zip")
    ns["MAX_ZIP_BYTES"], ns["MAX_ZIP_MEMBERS"] = 2_000_000_000, 2
    with pytest.raises(ValueError, match="the limits are"):
        extract(payload, root, "big.zip")
    assert not any(root.iterdir())
    ns["MAX_ZIP_MEMBERS"] = 20_000
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("records.jsonl", "")
        archive.writestr("../escaped.txt", "x")
    with pytest.raises(ValueError, match="would land outside the upload folder"):
        extract(buffer.getvalue(), root, "evil.zip")
    assert not any(root.iterdir()) and not (tmp_path / "escaped.txt").exists()
    with pytest.raises(ValueError, match="not a readable zip file"):
        extract(b"not a zip", root, "broken.zip")


def test_byod_provenance_replaces_the_corpus_block():
    source = _code_after("## 9. Re-extract the drawn chart")
    assert "'corpus': None if USE_BYOD else {" in source and "'byod': byod," in source


# --- DPC-m1: progress during long stages -----------------------------------------------------------


def _frozen_runner(image, max_new_tokens):
    """Reads the numbers but transposes the layout (one row per series instead of per category)."""
    return {"text": format_table([["North", "South"], ["10", "20"]]), "new_tokens": 9}


def _adapted_runner(image, max_new_tokens):
    return {"text": format_table([["North", "10"], ["South", "20"]]), "new_tokens": 9}


def _fixed_records(tmp_path: Path, n: int) -> list[dict]:
    out = []
    for r in _records(tmp_path, n):
        out.append({**r, "image": str(tmp_path / r["image"]), "target_text": format_table([["North", "10"], ["South", "20"]])})
    return out


def test_predict_and_evaluate_report_progress_after_every_chart(tmp_path):
    seen: list = []
    metrics = DePlotPipeline(_adapted_runner).evaluate(_fixed_records(tmp_path, 5), progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)] and metrics["n"] == 5


# --- DPC-M2: re-runs start from the pretrained model; adapt refuses an adapted pipeline ---------------


def test_adapt_refuses_an_already_adapted_pipeline(tmp_path):
    pipe = DePlotPipeline(_adapted_runner, adapter={"best_epoch": 1})
    with pytest.raises(ValueError, match="already adapted.*from_pretrained"):
        pipe.adapt(_fixed_records(tmp_path, 8))


def test_sections_5_to_7_reset_to_pretrained_and_section_6_refuses_an_adapted_model():
    for heading in ("## 5. Extract a table through the inference contract", "## 6. Baselines and the frozen model", "## 7. Bounded fine-tuning"):
        assert "reset_to_pretrained()" in _code_after(heading), heading
    section6 = _code_after("## 6. Baselines and the frozen model")
    assert section6.index("reset_to_pretrained()") < section6.index("pipe.evaluate(")
    assert "if frozen_test['adapted']:" in section6
    parity = _code_after("## 9. Re-extract the drawn chart")
    assert "raise RuntimeError(f'Reload parity failed: {parity}." in parity and "Re-run from Section 7" in parity


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline():
    source = _code_after("## 5. Extract a table through the inference contract")
    block = source[source.index("def reset_to_pretrained():") : source.index("def chart_progress(")]
    loads: list = []

    class _Loader:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return DePlotPipeline(_frozen_runner)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    ns = {"gc": __import__("gc"), "torch": fake_torch, "DePlotPipeline": _Loader, "WEIGHTS_DIR": "w", "pipe": DePlotPipeline(_frozen_runner)}
    exec(compile(block, "<reset>", "exec"), ns)
    ns["reset_to_pretrained"]()
    assert loads == []  # a pretrained pipeline is kept
    ns["pipe"] = DePlotPipeline(_adapted_runner, adapter={"best_epoch": 2})
    ns["reset_to_pretrained"]()
    assert loads == ["w"] and ns["pipe"].adapter is None


def test_experiments_name_their_field_and_the_cells_to_rerun():
    closing = _markdown().split("**Optional experiments (they do not affect the default path).**", 1)[1].split("## Troubleshooting", 1)[0]
    bullets = [line for line in closing.splitlines() if line.startswith("- **")]
    assert len(bullets) == 6
    for line in bullets[1:]:
        assert "Section" in line and "?" in line, line  # names the cell to change / re-run, and asks rather than tells
    assert "re-run from that cell" not in _markdown()


def _tiny_pipeline(seed: int = 0) -> DePlotPipeline:
    """A 3-layer, 32-wide Pix2Struct with the checkpoint's real vocabulary, processor and tokenizer."""
    from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, Pix2StructProcessor

    processor = Pix2StructProcessor.from_pretrained(str(DEFAULT_WEIGHTS_DIR), local_files_only=True)
    processor.image_processor.max_patches = 64
    config = Pix2StructConfig.from_pretrained(str(DEFAULT_WEIGHTS_DIR), local_files_only=True)
    text, vision = config.text_config, config.vision_config
    text.hidden_size, text.d_kv, text.num_heads, text.d_ff, text.num_layers = 32, 8, 4, 64, 3
    vision.hidden_size, vision.d_kv, vision.num_attention_heads, vision.d_ff = 32, 8, 4, 64
    vision.num_hidden_layers = 2
    config.text_config, config.vision_config = text, vision
    torch.manual_seed(seed)
    model = Pix2StructForConditionalGeneration(config)
    return DePlotPipeline._from_model(model, processor, "cpu", "tiny-random")


@needs_torch
def test_the_one_block_experiment_after_a_default_run_reproduces_its_model(tmp_path):
    """The review's journey on the tiny random model: a two-block run, then `TRAINABLE_DECODER_LAYERS = 1`.
    Re-adapting the same object is refused; the notebook's path (a fresh pretrained pipeline) trains from the base,
    starts epoch 0 from the frozen model, exports one block plus the final norm, and reloads with parity."""
    pytest.importorskip("transformers")
    records = _fixed_records(tmp_path, 12)
    first = _tiny_pipeline()
    first.adapt(records[:8], None, epochs=1, lr=1e-3, batch_size=4, trainable_decoder_layers=2)
    with pytest.raises(ValueError, match="already adapted"):
        first.adapt(records[:8], None, epochs=1, lr=1e-3, batch_size=4, trainable_decoder_layers=1)
    second = _tiny_pipeline()  # what reset_to_pretrained() does
    assert second.evaluate(records[8:], max_new_tokens=8)["adapted"] is False
    stages: list = []
    second.adapt(records[:8], records[8:], epochs=1, lr=1e-3, batch_size=4, trainable_decoder_layers=1, chart_progress=stages.append)
    assert {s["stage"] for s in stages} == {"train", "validation"}
    assert [s["charts"] for s in stages if s["stage"] == "train"] == [4, 8]
    assert [s["charts"] for s in stages if s["stage"] == "validation" and s["epoch"] == 0] == [1, 2, 3, 4]
    artifact = second.save_artifact(tmp_path / "adapter")
    tensors = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))["tensors"]
    assert {name.split(".")[2] for name in tensors if name.startswith("decoder.layer.")} == {"2"}
    reloaded = _tiny_pipeline()
    reloaded.load_artifact(artifact)
    before = [r["text"] for r in second.predict(records[8:], max_new_tokens=8)]
    after = [r["text"] for r in reloaded.predict(records[8:], max_new_tokens=8)]
    assert before == after
    for name, value in reloaded._model.state_dict().items():
        assert torch.equal(value, second._model.state_dict()[name]), name


# --- DPC-m3: Section 8 reads the three deltas together -------------------------------------------------


def _evaluation_namespace(tmp_path, monkeypatch, *, best_epoch: int) -> dict:
    """Sections 6 and 8 executed verbatim with injected runners (frozen: transposed layout; adapted: the target)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    records = _fixed_records(tmp_path, 24)
    splits = split_dataset(records, seed=42)
    progress: list = []
    ns: dict = {name: getattr(dcp, name) for name in dcp.__all__}
    ns.update(
        {
            "os": __import__("os"), "Path": Path, "json": json, "time": __import__("time"),
            "pipe": DePlotPipeline(_frozen_runner), "TABLE_MAX_TOKENS": 64, "USE_BYOD": True, "data_source": "BYOD (test.zip)",
            "train_records": splits["train"], "val_records": splits["validation"], "test_records": splits["test"],
            "dataset_manifests": {k: dcp.validate_dataset(v, min_records=1) for k, v in splits.items()},
            "disjoint": {k: len(v) for k, v in splits.items()}, "chart_types": {}, "reset_to_pretrained": lambda: None,
            "chart_progress": lambda label: (lambda done, total: progress.append((label, done, total))),
            "show_chart": lambda *a, **k: None, "show_table": lambda *a, **k: None, "_progress": progress,
        }
    )
    exec(compile(_code_after("## 6. Baselines and the frozen model"), "<section 6>", "exec"), ns)
    ns.update(
        {
            "pipe": DePlotPipeline(_adapted_runner, adapter={"best_epoch": best_epoch}),
            "EPOCHS": 3, "LEARNING_RATE": 1e-5, "TRAINABLE_DECODER_LAYERS": 2, "adapt_seconds": 1.0,
            "adapt_result": {"best_epoch": best_epoch, "selection": "highest validation cell accuracy", "history": [], "trainable_names": [], "n_trainable": 1},
        }
    )
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    return ns


def test_section8_names_a_layout_gain_and_prints_progress(tmp_path, monkeypatch, capsys):
    ns = _evaluation_namespace(tmp_path, monkeypatch, best_epoch=2)
    assert ns["frozen_test"]["adapted"] is False and ns["adapted_beats_frozen"] is True
    assert "cell accuracy rose" in ns["reading"] and "RNSS stayed flat" in ns["reading"] and "The gain is in layout" in ns["reading"]
    labels = {label for label, _done, _total in ns["_progress"]}
    assert labels == {"frozen model, test split", "adapted model, test split", "adapted model, validation split"}
    report = json.loads((tmp_path / "outputs" / "deplot_chart_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["reading"] == ns["reading"] and "exact_table_match" in report["comparison"]["delta_vs_frozen"]
    out = capsys.readouterr().out
    assert "Reading: " in out and "d_rnss" in out and len(ns["run_history"]) == 1


def test_section8_reports_a_kept_epoch_zero(tmp_path, monkeypatch):
    ns = _evaluation_namespace(tmp_path, monkeypatch, best_epoch=0)
    assert "The selector kept epoch 0" in ns["reading"]


def test_section6_refuses_an_adapted_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = _fixed_records(tmp_path, 12)
    splits = split_dataset(records, seed=42)
    ns: dict = {name: getattr(dcp, name) for name in dcp.__all__}
    ns.update({"time": __import__("time"), "pipe": DePlotPipeline(_adapted_runner, adapter={"best_epoch": 1}), "TABLE_MAX_TOKENS": 64, "train_records": splits["train"], "test_records": splits["test"], "reset_to_pretrained": lambda: None, "chart_progress": lambda label: None, "show_chart": lambda *a, **k: None, "show_table": lambda *a, **k: None})
    with pytest.raises(RuntimeError, match="must score the pretrained model"):
        exec(compile(_code_after("## 6. Baselines and the frozen model"), "<section 6>", "exec"), ns)


def test_no_learner_cell_asserts_a_result():
    for cell in _cells():
        source = _source(cell)
        if cell["cell_type"] != "code" or "dimer" in cell.get("metadata", {}) or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:120]


# --- DPC-m4: the charts are shown -------------------------------------------------------------------


def test_the_drawn_chart_and_one_chart_per_type_are_displayed():
    assert "show_chart(image)" in _code_after("## 5. Extract a table through the inference contract")
    section6 = _code_after("## 6. Baselines and the frozen model")
    assert "for chart_type in sorted(frozen_test['by_chart_type']):" in section6 and "show_chart(test_by_id[row['id']]['image'], width=360)" in section6
    assert "show_table('adapted model', after['prediction'])" in _code_after("## 8. Held-out evaluation")


# --- DPC-M1: isolated runtime ---------------------------------------------------------------------------


def test_exactly_two_kernel_cells_and_no_in_kernel_pip_install():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 2
    install = _source(kernel[0])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install
    markdown = _markdown()
    assert "Restart the runtime" not in markdown and "an interpreter restart after the install is expected" not in markdown


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on a spec-less stub."""
    import ast
    import importlib.util

    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)  # raised ValueError on a spec-less stub
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


# --- DPC-M3: guided layer and infrastructure labels ------------------------------------------------------


def test_guided_layer_and_infrastructure_labels():
    markdown = _markdown()
    for marker, least in (("**Predict before running:**", 6), ("**What to notice:**", 6), ("<summary>Check your reasoning</summary>", 7), ("> **Infrastructure.**", 3)):
        assert markdown.count(marker) >= least, marker
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## 10. Your turn — change one thing", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert marker in markdown, marker
    code = [c for c in _cells() if c["cell_type"] == "code"]
    setup = code[:7]  # install, router, runtime record, three carried modules, model staging
    assert all(c["metadata"].get("cellView") == "form" for c in setup)
    titled = [c for c in setup if "embedded_module" not in c["metadata"].get("dimer", {})]
    assert len(titled) == 4 and all(_source(c).startswith("# @title Infrastructure: ") for c in titled)
    assert "cellView" not in code[7]["metadata"]  # the learning path starts in Section 4


def test_durations_name_their_environment():
    markdown = _markdown()
    assert "each extraction costs seconds on CPU" not in markdown
    for figure in ("4,794 s", "2,290 s"):
        for match in re.finditer(re.escape(figure), markdown):
            window = markdown[max(0, match.start() - 300) : match.end() + 120]
            assert "T4" in window, figure
    assert "2.4 hours" in markdown and "estimate" in markdown
