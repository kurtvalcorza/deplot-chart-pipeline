"""Chart-to-table extraction with the pinned ``google/deplot`` checkpoint (DePlot, a Pix2Struct model).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Pix2Struct architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed. The fixed instruction is rendered
as a text header on top of the chart (the Pix2Struct VQA input convention) with Pillow's bundled font,
so no font is fetched from the Hub at inference time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

MODEL_ID = "google/deplot"
MODEL_REVISION = "6e76d62430da16986be3426bae32301fb9115397"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "deplot"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

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
    """``_runner(image, max_new_tokens)`` returns ``{"text": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

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
        font_bytes = header_font_bytes()
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Pix2StructProcessor.from_pretrained(location, **common)
        if not getattr(processor.image_processor, "is_vqa", False):
            raise RuntimeError("snapshot image processor is not the VQA variant (is_vqa=False); refusing")
        model = Pix2StructForConditionalGeneration.from_pretrained(location, dtype=torch.float32, **common)
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, max_new_tokens: int) -> dict[str, Any]:
            # The image processor is called directly: Pix2StructProcessor.__call__ drops the
            # font_bytes kwarg, and font_bytes is what replaces the default Hub font download
            # (see header_font_bytes). The VQA processor renders the instruction as the header.
            inputs = processor.image_processor(
                image, header_text=INSTRUCTION, return_tensors="pt", font_bytes=font_bytes
            ).to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return {"text": decoded, "new_tokens": int(generated[0].shape[0]) - 1}

        return cls(runner, resolved_device, "float32", source)

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
