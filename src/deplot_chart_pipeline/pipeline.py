"""Chart-to-table extraction with the pinned ``google/deplot`` checkpoint (DePlot, a Pix2Struct model).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Pix2Struct architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed. The fixed instruction is rendered
as a text header on top of the chart (the Pix2Struct VQA input convention) with Pillow's bundled font,
so no font is fetched from the Hub at inference time.

The adaptation contract (`predict`, `evaluate`, `adapt`, `save_artifact`, `load_artifact`, `from_artifact`)
fine-tunes the decoder's last blocks on validated chart/table records with the linearised table as the
target, selects the epoch on validation cell accuracy and exports the trained tensors as a safetensors
adapter bound to the pinned base.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

MODEL_ID = "google/deplot"
MODEL_REVISION = "6e76d62430da16986be3426bae32301fb9115397"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "deplot"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = "ab90055611f42fee327d9ecf3c9cdac63e847bd19a0ac8ea86b0e8134fe0711b"
PARAMETER_COUNT = 282_285_696  # 18,879,744 of them train by default (2 decoder blocks + final norm)
DECODER_LAYERS = 12
DEFAULT_TRAINABLE_DECODER_LAYERS = 2
# Teacher-forcing ceiling for a target table (the SynthChartNet sample's longest target is 360 tokens).
MAX_TARGET_TOKENS = 512
# Evaluation bounds: a split larger than MAX_EVAL_RECORDS is refused (every chart is one generation of up
# to max_new_tokens); below MIN_SCORED_RECORDS the verdict says the sample is small.
MAX_EVAL_RECORDS = 500
MIN_SCORED_RECORDS = 20
ARTIFACT_FORMAT = "org.valcorza.deplot.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"

# The instruction the pinned README renders above the chart; DePlot was trained on this exact prompt
# and the pipeline does not expose any other.
INSTRUCTION = "Generate underlying data table of the figure below:"
# Generation ceilings. 512 is the max_new_tokens the pinned README's example passes; the ceiling
# leaves room for a long table.
MAX_NEW_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 512
DECODING = "greedy"
# Output conventions: DePlot linearises a table as rows separated by the literal token ``<0x0A>`` and
# cells separated by ``|``; the first row is ``TITLE | <chart title>`` when a title was read.
ROW_SEPARATOR = "<0x0A>"
CELL_SEPARATOR = "|"
# Input ceilings. The processor extracts at most MAX_PATCHES 16x16 patches (preprocessor_config.json)
# after scaling the image to fill that budget, so pixel count only guards memory during resizing.
MAX_PATCHES = 2048
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
# Relaxed numeric match tolerance for cell_accuracy (the ChartQA/DePlot "relaxed accuracy" convention).
RELATIVE_TOLERANCE = 0.05
_NUMBER_RE = re.compile(r"^[-+]?\$?\s*(\d[\d,]*\.?\d*|\.\d+)\s*%?$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def header_font_bytes() -> bytes:
    """Pillow's bundled Aileron Regular (CC0) as TrueType bytes: the header font for the rendered question.

    The upstream image processor otherwise fetches ``ybelkada/fonts/Arial.TTF`` from the Hub at
    inference time — an unpinned, unlisted download of a proprietary font. The bundled subset covers
    the printable ASCII range, which is what a question is expected to use.
    """
    font = ImageFont.load_default(size=36)
    data = getattr(font, "font_bytes", None)
    if not data:
        raise RuntimeError("Pillow's bundled TrueType font is unavailable (FreeType support missing)")
    return bytes(data)


def parse_table(text: str) -> dict[str, Any]:
    """Split DePlot's linearised output into a title (or None) and a list of rows of stripped cells."""
    rows: list[list[str]] = []
    title: str | None = None
    for raw_row in text.split(ROW_SEPARATOR):
        cells = [cell.strip() for cell in raw_row.split(CELL_SEPARATOR)]
        if not any(cells):
            continue
        if title is None and not rows and len(cells) >= 2 and cells[0].upper() == "TITLE":
            title = CELL_SEPARATOR.join(cells[1:]).strip()
            continue
        rows.append(cells)
    return {
        "title": title,
        "rows": rows,
        "n_rows": len(rows),
        "n_columns": max((len(r) for r in rows), default=0),
    }


def _as_number(cell: str) -> float | None:
    match = _NUMBER_RE.match(cell.strip())
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def cells_match(predicted: str, expected: str, *, relative_tolerance: float = RELATIVE_TOLERANCE) -> bool:
    """Relaxed cell match: equal after case/whitespace normalisation, or numerically within the tolerance."""
    if " ".join(predicted.lower().split()) == " ".join(expected.lower().split()):
        return True
    p, e = _as_number(predicted), _as_number(expected)
    if p is None or e is None:
        return False
    return abs(p - e) <= relative_tolerance * abs(e) if e != 0 else abs(p) <= relative_tolerance


def cell_accuracy(
    predicted_rows: Sequence[Sequence[str]],
    expected_rows: Sequence[Sequence[str]],
    *,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Position-wise relaxed cell accuracy of a predicted table against the expected one.

    Every expected cell (row i, column j) counts once; it is matched only against the predicted cell at
    the same position (a missing row or column is a miss, an extra one is not penalised here but is
    reported through the shape fields). This is a sanity measure, not the paper's RMS metric.
    """
    if not expected_rows or not any(expected_rows):
        raise ValueError("expected_rows must contain at least one cell")
    total = matched = 0
    for i, expected in enumerate(expected_rows):
        predicted = predicted_rows[i] if i < len(predicted_rows) else []
        for j, cell in enumerate(expected):
            total += 1
            if j < len(predicted) and cells_match(predicted[j], cell, relative_tolerance=relative_tolerance):
                matched += 1
    return {
        "matched": matched,
        "total": total,
        "value": matched / total,
        "predicted_shape": [len(predicted_rows), max((len(r) for r in predicted_rows), default=0)],
        "expected_shape": [len(expected_rows), max((len(r) for r in expected_rows), default=0)],
        "relative_tolerance": relative_tolerance,
    }


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one chart image as PIL.Image.Image (any mode, converted to RGB): a bar, line or pie chart",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "instruction": INSTRUCTION,
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False), deterministic on a fixed device and dtype",
    "preprocessing": (
        "the fixed instruction is rendered as a black-on-white header (Pillow's bundled font) above the "
        "chart; the composite is scaled to fill at most MAX_PATCHES 16x16 patches (aspect ratio preserved), "
        "normalised per image, and flattened into patch tokens with row/column positions; the decoder "
        "generates the linearised table"
    ),
    "output": (
        f"linearised table text (rows separated by {ROW_SEPARATOR!r}, cells by {CELL_SEPARATOR!r}, optional "
        "leading TITLE row) plus its parsed rows; no score"
    ),
}


def _check_inputs(image: Any, max_new_tokens: Any) -> tuple[Image.Image, int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``extract_table`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, max_new_tokens


def validate_inputs(
    image: Image.Image,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``extract_table`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, checked_tokens = _check_inputs(image, max_new_tokens)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (extract_table takes one chart image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "instruction": INSTRUCTION,
        "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    expected_rows: Sequence[Sequence[str]] | None = None,
    *,
    expected_title: str | None = None,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``expected_rows`` (the table the chart really encodes, header row first) the report carries
    position-wise relaxed ``cell_accuracy`` and, when ``expected_title`` is given, a ``title_match``
    entry, verdict ``sample-sanity``; without ``expected_rows`` it is ``not-measurable`` and says what
    labelled data would make the task measurable.
    """
    table = result.get("table") or parse_table(str(result["text"]))
    base = {
        "task": "chart image -> linearised data table (plot-to-table)",
        "score_semantics": (
            "the table is generated text and carries no score, probability or correctness signal; a "
            "well-formed table is not evidence that its numbers are read from the chart. Greedy decoding "
            "makes the output reproducible on a fixed device and dtype, a reproducibility property, not a "
            "quality one"
        ),
        "sample_kind": sample_kind,
        "predicted_shape": [table["n_rows"], table["n_columns"]],
        "truncated": result.get("truncated"),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if expected_rows is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no expected data table was supplied for the evaluated chart",
            "needs": (
                "chart images paired with their underlying data tables from the deployment domain (chart "
                "types, styles, renderers) scored with relaxed cell accuracy or the DePlot paper's relative "
                "mapping similarity; no such labelled set ships with this repository"
            ),
        }
    accuracy = cell_accuracy(table["rows"], expected_rows)
    metrics: list[dict[str, Any]] = [
        {
            "id": "cell_accuracy",
            "value": accuracy["value"],
            "matched": accuracy["matched"],
            "total": accuracy["total"],
            "predicted_shape": accuracy["predicted_shape"],
            "expected_shape": accuracy["expected_shape"],
            "normalisation": (
                "position-wise; text cells compared case/whitespace-insensitively, numeric cells within "
                f"{RELATIVE_TOLERANCE:.0%} relative tolerance"
            ),
            "estimation": "one chart, no dispersion estimate",
        }
    ]
    if expected_title is not None:
        metrics.append(
            {
                "id": "title_match",
                "value": 1.0 if table["title"] and cells_match(table["title"], expected_title) else 0.0,
                "predicted": table["title"],
                "expected": expected_title,
                "estimation": "one chart, structural sanity only",
            }
        )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} sanity measure(s) on one tutorial chart whose data you rendered yourself; "
            "plumbing evidence, not a chart-to-table benchmark"
        ),
        "needs": (
            "a labelled chart/table set from the deployment domain (chart types, styles, renderers, "
            "languages) for any plot-to-table accuracy claim"
        ),
    }


@dataclass
class DePlotPipeline:
    """``_runner(image, max_new_tokens)`` returns ``{"text": str, "new_tokens": int}``; injectable so the
    offline tests run without the model."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    _font_bytes: bytes | None = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> DePlotPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        header_font_bytes()
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Pix2StructProcessor.from_pretrained(location, **common)
        if not getattr(processor.image_processor, "is_vqa", False):
            raise RuntimeError("snapshot image processor is not the VQA variant (is_vqa=False); refusing")
        # The model is loaded in float32 for CPU inference and training.
        model = Pix2StructForConditionalGeneration.from_pretrained(location, dtype=torch.float32, **common)
        return cls._from_model(model, processor, resolved_device, source)

    @classmethod
    def _from_model(cls, model: Any, processor: Any, device: str, source: str) -> DePlotPipeline:
        """Wrap a constructed model and VQA processor (every parameter frozen, eval mode) in a pipeline; the
        offline tests use it with a small randomly initialised Pix2Struct model."""
        import torch

        font_bytes = header_font_bytes()
        model = model.eval().to(device)
        for param in model.parameters():
            param.requires_grad_(False)

        def runner(image: Image.Image, max_new_tokens: int) -> dict[str, Any]:
            # The image processor is called directly: Pix2StructProcessor.__call__ drops the
            # font_bytes kwarg, and font_bytes is what replaces the default Hub font download
            # (see header_font_bytes). The VQA processor renders the instruction as the header.
            inputs = processor.image_processor(
                image, header_text=INSTRUCTION, return_tensors="pt", font_bytes=font_bytes
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
                )
            # Encoder-decoder: the output holds only decoder tokens (decoder_start + table + eos).
            decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return {"text": decoded, "new_tokens": int(generated[0].shape[0]) - 1}

        return cls(
            runner, device, "float32", source, _model=model, _processor=processor, _font_bytes=font_bytes
        )

    def extract_table(
        self,
        image: Image.Image,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Translate one chart image into its linearised data table; ``table`` is the parsed form."""
        rgb, checked_tokens = _check_inputs(image, max_new_tokens)
        raw = self._runner(rgb, checked_tokens)
        if not isinstance(raw, dict) or "text" not in raw:
            raise RuntimeError("runner must return a dict with 'text'")
        text = str(raw["text"]).strip()
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "text": text,
            "table": parse_table(text),
            "instruction": INSTRUCTION,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- adaptation contract -----------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._processor

    def predict(
        self, records: Sequence[Mapping[str, Any]], *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> list[dict[str, Any]]:
        """Extract the table of every validated record's chart; one `extract_table` result per record, in
        order, with the record's `id` and `chart_type` attached."""
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        out = []
        for record in checked:
            with Image.open(record["image"]) as image:
                image.load()
                result = self.extract_table(image, max_new_tokens=max_new_tokens)
            out.append({"id": record["id"], "chart_type": record["chart_type"], **result})
        return out

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> dict[str, Any]:
        """Extract every record's table and score it against the record's target: mean relaxed position-wise
        cell accuracy, RNSS, exact-table match, the truncation rate and a per-chart-type breakdown (see
        `metrics.chart_metrics`)."""
        from .metrics import chart_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        results = self.predict(checked, max_new_tokens=max_new_tokens)
        metrics = chart_metrics([r["text"] for r in results], checked)
        metrics.update(
            {
                "truncated_rate": sum(bool(r["truncated"]) for r in results) / len(results),
                "max_new_tokens": max_new_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _decoder_layers(self) -> int:
        model, _ = self._require_model()
        return int(model.config.text_config.num_layers)

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        """The last `trainable_decoder_layers` blocks of the text decoder plus the decoder's final layer norm.
        The untied output projection (`decoder.lm_head`, vocabulary x hidden) and every embedding stay frozen,
        as does the whole image encoder."""
        if (
            isinstance(trainable_decoder_layers, bool)
            or not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        n_layers = self._decoder_layers()
        if trainable_decoder_layers > n_layers:
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{n_layers} for this model")
        first = n_layers - trainable_decoder_layers
        prefixes = tuple(f"decoder.layer.{k}." for k in range(first, n_layers))
        prefixes += ("decoder.final_layer_norm.",)
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def _encode_batch(self, records: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
        """The frozen encoder's output and patch mask for a batch of (chart, rendered instruction) inputs.
        The header is part of the image, so every chart is its own encoder input; it is recomputed per step
        under `no_grad` instead of cached (2,048 x 768 floats per chart)."""
        import torch

        model, processor = self._require_model()
        device = next(model.parameters()).device
        images = []
        for record in records:
            with Image.open(record["image"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor.image_processor(
            images, header_text=[INSTRUCTION] * len(images), return_tensors="pt", font_bytes=self._font_bytes
        ).to(device)
        with torch.no_grad():
            hidden = model.encoder(
                flattened_patches=inputs["flattened_patches"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state
        return hidden, inputs["attention_mask"]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 3,
        lr: float = 1e-5,
        batch_size: int = 4,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on validated chart/table records.

        Only the last `trainable_decoder_layers` blocks of the text decoder and the decoder's final layer norm
        train (2 blocks by default); the image encoder, every embedding and the untied output projection stay
        frozen. Each record is one training sample: the fixed instruction is rendered above the chart exactly
        as `extract_table` renders it, the frozen encoder reads the composite, and the target is the tokenised
        linearised table (`target_text`, at most MAX_TARGET_TOKENS tokens) with its end-of-sequence token,
        decoded with teacher forcing and scored with the model's own cross-entropy (padding ignored); AdamW at
        a fixed learning rate with gradient clipping at 1.0, no scheduler. Epoch 0 records the frozen model's
        validation metrics; the epoch with the highest validation cell accuracy is kept (ties keep the
        earlier)."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        import torch

        torch.manual_seed(seed)
        model, processor = self._require_model()
        tokenizer = processor.tokenizer
        pad_id = int(tokenizer.pad_token_id)
        started = time.perf_counter()
        wanted = set(names)
        original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
        original_training = model.training
        original_adapter = copy.deepcopy(self.adapter)
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked).items()
                if k in ("cell_accuracy", "rnss", "exact_table_match", "n")
            }

        try:
            for name, param in model.named_parameters():
                param.requires_grad_(name in wanted)
            params = [p for p in model.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in params)
            optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            device = next(model.parameters()).device
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {
                "epoch": 0,
                "train_loss": None,
                "val": score_val(),
                "note": "frozen model",
            }
            history.append(entry)
            if progress:
                progress(entry)
            best_score = entry["val"]["cell_accuracy"] if entry["val"] else -math.inf
            best_state = {k: v.clone() for k, v in initial_state.items()}
            best_epoch = 0
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    chosen = [train_checked[j] for j in order[start : start + batch_size]]
                    hidden, mask = self._encode_batch(chosen)
                    targets = tokenizer(
                        [r["target_text"] for r in chosen],
                        padding=True,
                        truncation=True,
                        max_length=MAX_TARGET_TOKENS,
                        return_tensors="pt",
                    ).to(device)
                    labels = targets["input_ids"].masked_fill(targets["input_ids"] == pad_id, -100)
                    out = model(encoder_outputs=(hidden,), attention_mask=mask, labels=labels)
                    optimiser.zero_grad(set_to_none=True)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(out.loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["cell_accuracy"] if entry["val"] else math.inf
                if current > best_score or not entry["val"]:
                    best_score = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, including its mode, gradient flags and any previously loaded adapter.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.train(original_training)
            for name, param in model.named_parameters():
                param.requires_grad_(original_requires_grad[name])
            self.adapter = original_adapter
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": (
                "highest validation cell accuracy" if val_checked else "final epoch (no validation split)"
            ),
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ---------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
            if name in names
        }
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {
                key: value
                for key, value in self.adapter.items()
                if key not in ("history", "trainable_names")
            },
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError("artifact format_version is not supported")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file") != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_decoder_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= DECODER_LAYERS:
            raise ValueError("artifact manifest does not record valid trainable decoder layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_decoder_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("decoder."):
                raise ValueError(f"artifact tensor {key} is not an adaptable decoder tensor of the base")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({key: value.to(state[key].dtype) for key, value in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> DePlotPipeline:
        pipeline = cls.from_pretrained(
            device=device,
            weights_dir=weights_dir,
            allow_download=allow_download,
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
