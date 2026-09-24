"""Adaptation checks on a real Pix2Struct network: a small randomly initialised model built offline from the
committed snapshot config, VQA processor and tokenizer with the header rendered in Pillow's bundled font
(always), and the pinned checkpoint itself where its weights are staged (local pre-flight). They cover a
scored evaluation, a one-epoch adaptation of the last decoder block, the trainable-tensor scope, the artifact
round trip, the loader's tensor-set check, the transactional guarantee and — where CUDA is visible — the same
path on the accelerator."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deplot_chart_pipeline import (  # noqa: E402  -- after the importorskip guards
    DEFAULT_WEIGHTS_DIR,
    WEIGHT_FILE,
    DePlotPipeline,
    format_table,
)

LABELS = ["North", "South", "East", "West"]
REAL_ONE_BLOCK_TRAINABLE = 9_440_256  # decoder.layer.11.* (9,439,488) + decoder.final_layer_norm (768)
TOKENS = 24  # generation budget for the checks; the tables below are a few rows long


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


def _real_pipeline(device: str = "cpu") -> DePlotPipeline:
    return DePlotPipeline.from_pretrained(device=device)


def _real_staged() -> bool:
    return (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file()


def _chart(path, values):
    image = Image.new("RGB", (256, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(24, 8), (24, 168), (248, 168)], fill="black", width=2)
    for k, (label, value) in enumerate(zip(LABELS, values, strict=True)):
        x0 = 36 + 52 * k
        draw.rectangle([x0, 168 - 12 * value, x0 + 32, 167], fill=(60, 110, 200))
        draw.text((x0 + 10, 168 - 12 * value - 12), str(value), fill="black")
        draw.text((x0, 174), label, fill="black")
    image.save(path, format="PNG")
    return path


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("charts")
    out = []
    for i in range(12):
        values = [1 + (i + k) % 9 for k in range(4)]
        path = _chart(root / f"chart{i}.png", values)
        out.append(
            {
                "id": f"c{i:02d}",
                "image_id": f"chart{i}",
                "image": str(path),
                "target_text": format_table(
                    [[label, str(v)] for label, v in zip(LABELS, values, strict=True)]
                ),
                "chart_type": "bar",
            }
        )
    return out


def _tables(pipe, records):
    out = []
    for r in records:
        with Image.open(r["image"]) as image:
            image.load()
            out.append(pipe.extract_table(image, max_new_tokens=TOKENS)["text"])
    return out


@pytest.fixture(params=["tiny", "real"])
def kind(request):
    if request.param == "real" and not _real_staged():
        pytest.skip("pinned snapshot not staged")
    return request.param


@pytest.fixture
def pipe(kind):
    return _tiny_pipeline() if kind == "tiny" else _real_pipeline()


def _fresh(kind):
    return _tiny_pipeline() if kind == "tiny" else _real_pipeline()


def _load(kind, artifact):
    fresh = _fresh(kind)
    fresh.load_artifact(artifact)
    return fresh


def test_evaluate_scores_tables(pipe, records):
    metrics = pipe.evaluate(records[:5], max_new_tokens=TOKENS)
    assert metrics["n"] == 5 and 0.0 <= metrics["cell_accuracy"] <= 1.0 and metrics["adapted"] is False
    assert 0.0 <= metrics["rnss"] <= 1.0 and set(metrics["by_chart_type"]) == {"bar"}
    assert set(metrics) >= {"exact_table_match", "truncated_rate", "verdict", "seconds"}


def test_one_epoch_adaptation_scope_and_artifact_round_trip(kind, pipe, records, tmp_path):
    result = pipe.adapt(records[:8], records[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    last = pipe._model.config.text_config.num_layers - 1
    assert all(
        name.startswith(f"decoder.layer.{last}.") or name.startswith("decoder.final_layer_norm.")
        for name in result["trainable_names"]
    )
    assert "decoder.lm_head.weight" not in result["trainable_names"]
    assert not any(name.startswith("encoder.") for name in result["trainable_names"])
    if kind == "real":
        assert result["n_trainable"] == REAL_ONE_BLOCK_TRAINABLE
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"cell_accuracy", "rnss", "exact_table_match", "n"}
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    assert manifest["base_model"]["weight_file"] == WEIGHT_FILE
    reloaded = _load(kind, artifact)
    assert _tables(pipe, records[:4]) == _tables(reloaded, records[:4])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_training_moves_the_trainable_tensors_only(kind, pipe, records):
    if kind == "real":
        pytest.skip("the tiny model is enough to check what moves")
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    result = pipe.adapt(records[:8], None, epochs=2, lr=1e-3, trainable_decoder_layers=1, batch_size=4)
    after = pipe._model.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and moved <= set(result["trainable_names"])
    assert result["history"][2]["train_loss"] < result["history"][1]["train_loss"]


def test_no_validation_keeps_the_final_epoch_and_reloads_it(kind, pipe, records, tmp_path):
    result = pipe.adapt(records[:8], None, epochs=2, trainable_decoder_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = _load(kind, artifact)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 2 and reloaded.adapter["trainable_decoder_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(
    kind, pipe, records, tmp_path
):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:8], None, epochs=1, trainable_decoder_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        _load(kind, fewer)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [
        {**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}
    ]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        _load(kind, extra)
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_decoder_layers": 2}
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        _load(kind, other_layers)
    tampered = tmp_path / "tampered"
    shutil.copytree(artifact, tampered)
    (tampered / "adapter.safetensors").write_bytes(
        (tampered / "adapter.safetensors").read_bytes()[:-1] + b"\0"
    )
    with pytest.raises(ValueError, match="digest or size mismatch"):
        _load(kind, tampered)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:8], None, epochs=2, trainable_decoder_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_extract_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    """Every tensor the runner, the encoder pass and the trainer build must land on the model's device."""
    if not _real_staged():
        pytest.skip("pinned snapshot not staged")
    cuda = _real_pipeline(device="cuda:0")
    assert cuda.device == "cuda:0"
    with Image.open(records[0]["image"]) as image:
        image.load()
        first = cuda.extract_table(image, max_new_tokens=TOKENS)
    assert first["device"] == "cuda:0" and isinstance(first["text"], str)
    result = cuda.adapt(records[:8], records[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = DePlotPipeline.from_artifact(artifact, device="cuda:0")
    assert _tables(cuda, records[:4]) == _tables(reloaded, records[:4])
