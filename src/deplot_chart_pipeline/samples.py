"""Chart-to-table dataset contract for fine-tuning: the pinned SynthChartNet sample, the OTSL-to-DePlot target
converter, validation, seeded chart-type-stratified splitting, BYOD loaders and JSONL export.

The default dataset is **real** chart/table pairs: charts from `docling-project/SynthChartNet` (bar, pie,
stacked-bar and line charts rendered with Matplotlib, Seaborn and Pyecharts, each paired with the data table
it was drawn from, published under **CDLA-Permissive-2.0**). SynthChartNet is not part of DePlot's
fine-tuning mixture (synthetic, ChartQA and PlotQA plots), so this is adaptation to a new chart family, not a
re-run of the checkpoint's own data. The sample is one pinned parquet shard (`train-00000-of-00135.parquet`,
515,825,444 bytes) downloaded whole at the pinned dataset revision and refused unless its SHA-256 matches
the pin before `pyarrow` reads a byte of it. A seeded, chart-type-stratified subset of charts is drawn from
it and cut into training, validation and test charts; every chart image is written to the cache under its
own content digest, and a chart whose image digest was already drawn is skipped, so no image is shared
between splits.

**Target format — the model's own convention, not an assumption.** SynthChartNet stores each table as OTSL
(``<fcel>`` a cell, ``<ecel>`` an empty cell, ``<nl>`` a row end, after ``<loc_…>`` box tokens and a
``<{bar|pie|stacked_bar|line}_chart>`` type tag). The pinned DePlot checkpoint, run frozen on bar, pie,
stacked-bar and line charts of this shard, always emits ``TITLE | <title>`` first (the title empty: these
charts carry none), then one row per **axis category / x point** with the series as columns, cells joined by
`` | `` and rows by `` <0x0A> ``; it writes a header row (empty corner, series names) when the chart has a
legend or a series name and none for a single unnamed series (pies). OTSL orients bar, pie and stacked-bar
tables the other way — the axis categories run along the first row and each further row is one series — while
line tables already hold one row per x point under a ``<ecel> | series`` header. The converter
(`otsl_to_rows`, `target_from_otsl`) therefore applies **one rule**:

* bar, pie, stacked_bar: transpose the OTSL grid; line: keep it;
* ``<ecel>`` becomes an empty cell; nothing else is dropped, merged or renamed;
* the target is ``TITLE | `` followed by the oriented rows, cells joined by `` | `` and rows by `` <0x0A> ``.

A two-row single-series bar or pie table thus becomes ``category | value`` rows with no header (it has no
series name), and a table with an ``<ecel>`` corner keeps a header row ``  | series …``. The literal
``<0x0A>`` in a target tokenises to the same byte token the model emits.

A record is ``{id, image_id, image, target_text, chart_type}`` — the path of the chart image, the linearised
target table, and the chart type (`bar`, `pie`, `stacked_bar`, `line`; BYOD records default to `other`).

The shard's SHA-256 and row count are recorded by `tools/pin_corpus.py`; until they are recorded,
`fetch_corpus` refuses to read the shard rather than read an unpinned file.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import CELL_SEPARATOR, MODEL_ID, ROW_SEPARATOR, parse_table, validate_image

CORPUS_NAME = "SynthChartNet"
CORPUS_REPO = "docling-project/SynthChartNet"
CORPUS_REVISION = "b913ef98a3d45f6136465963ddc71a7a6b0e1728"
CORPUS_RELEASE = (
    "SynthChartNet training shard 0 of 135 (14,568 charts with their OTSL data tables) on the Hugging Face "
    "Hub, dataset revision b913ef98"
)
CORPUS_LICENSE = (
    "CDLA-Permissive-2.0 (declared in the Hub dataset card): use, modification and sharing permitted, "
    "no restriction on results such as trained weights; attribution to SynthChartNet (docling-project) "
    "required"
)
CORPUS_COLUMNS = ("images", "texts")
# The shard's pins. `sha256` and `rows` are written by tools/pin_corpus.py from a verified file; while
# `sha256` is None the reader refuses to run.
CORPUS_FILE: dict[str, Any] = {
    "path": "train-00000-of-00135.parquet",
    "bytes": 515_825_444,
    "sha256": "1d6cf578fdd953fa0dd7c1d169926eb3418daa2770cc3a301b1a64bec0b209e7",
    "rows": 14568,
}
DEFAULT_CACHE_DIR = Path("weights") / "synthchartnet"
SAMPLE_SEED = 42
# Chart counts per split; each split is stratified by chart type in proportion to the shard's type mix
# (largest remainder), so a rare type can receive no chart in a small split.
SAMPLE_CHARTS = {"train": 360, "validation": 80, "test": 160}
CHART_TYPES = ("bar", "pie", "stacked_bar", "line")
# OTSL orients these types with the axis categories along the first row; DePlot emits one row per category.
TRANSPOSED_TYPES = frozenset({"bar", "pie", "stacked_bar"})
MIN_RECORDS = 8
MAX_RECORDS = 5_000
# Target ceiling: the linearised table is also the decoder's teacher-forcing target, so it is bounded to
# keep it well inside MAX_TARGET_TOKENS of the pipeline (the shard's longest target under it is 360 tokens).
MAX_TARGET_CHARS = 512
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_TYPE_RE = re.compile(r"<(bar|pie|stacked_bar|line)_chart>")
_CELL_RE = re.compile(r"<(fcel|ecel)>([^<]*)")
_TITLE_ROW = f"TITLE {CELL_SEPARATOR} "
_ROW_JOIN = f" {ROW_SEPARATOR} "
_CELL_JOIN = f" {CELL_SEPARATOR} "


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- the OTSL -> DePlot target converter ------------------------------------------------------


def parse_otsl(otsl: str) -> dict[str, Any]:
    """The chart type and the OTSL grid (rows of cell strings, ``<ecel>`` as ``""``) of one SynthChartNet
    table; raises ValueError on a missing type tag, an unknown tag, an empty or a ragged grid."""
    if not isinstance(otsl, str):
        raise ValueError("OTSL table must be a string")
    match = _TYPE_RE.search(otsl)
    if not match:
        raise ValueError("OTSL table has no <{bar|pie|stacked_bar|line}_chart> tag")
    body = otsl[match.end() :].replace("</chart>", "")
    grid: list[list[str]] = []
    for raw_row in body.split("<nl>"):
        if not raw_row.strip():
            continue
        cells = _CELL_RE.findall(raw_row)
        if "".join(f"<{kind}>{text}" for kind, text in cells) != raw_row:
            raise ValueError(f"OTSL row holds a tag other than <fcel>/<ecel>: {raw_row[:60]!r}")
        grid.append([" ".join(text.split()) if kind == "fcel" else "" for kind, text in cells])
    if not grid:
        raise ValueError("OTSL table has no rows")
    if len({len(row) for row in grid}) != 1:
        raise ValueError("OTSL table is ragged (rows of different widths)")
    if any(CELL_SEPARATOR in cell or ROW_SEPARATOR in cell for row in grid for cell in row):
        raise ValueError("OTSL cell contains a DePlot separator")
    return {"chart_type": match.group(1), "rows": grid}


def otsl_to_rows(otsl: str) -> list[list[str]]:
    """The OTSL grid oriented as DePlot emits it: transposed for bar, pie and stacked-bar charts (one row per
    axis category), unchanged for line charts (already one row per x point)."""
    parsed = parse_otsl(otsl)
    grid = parsed["rows"]
    if parsed["chart_type"] in TRANSPOSED_TYPES:
        grid = [list(column) for column in zip(*grid, strict=True)]
    return grid


def format_table(rows: Sequence[Sequence[str]], title: str = "") -> str:
    """DePlot's linearisation: ``TITLE | <title>``, then the rows; cells joined by `` | ``, rows by
    `` <0x0A> ``."""
    return _ROW_JOIN.join([_TITLE_ROW + title, *(_CELL_JOIN.join(row) for row in rows)])


def target_from_otsl(otsl: str) -> str:
    """The training/evaluation target of one SynthChartNet table (see the module docstring for the rule)."""
    return format_table(otsl_to_rows(otsl))


# ---- the pinned shard ----


def corpus_pinned() -> bool:
    """Whether the shard's SHA-256 has been recorded (see tools/pin_corpus.py)."""
    return isinstance(CORPUS_FILE.get("sha256"), str) and len(CORPUS_FILE["sha256"]) == 64


def _hub_download(cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            CORPUS_REPO,
            CORPUS_FILE["path"],
            repo_type="dataset",
            revision=CORPUS_REVISION,
            local_dir=str(cache),
        )
    )


def fetch_corpus(
    *, cache_dir: str | Path | None = None, downloader: Callable[[Path], Path] | None = None
) -> Path:
    """Return the path of the pinned shard, downloading it at the pinned revision when the cached copy is
    absent or drifted; refused on any size or SHA-256 mismatch, and outright while no pin is recorded."""
    if not corpus_pinned():
        raise RuntimeError(
            f"{CORPUS_REPO}@{CORPUS_REVISION[:8]} {CORPUS_FILE['path']}: no SHA-256 pin is recorded; run "
            "tools/pin_corpus.py with Hub access to record it before the sample can be read"
        )
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / CORPUS_FILE["path"]

    def ok(path: Path) -> bool:
        return (
            path.is_file()
            and path.stat().st_size == CORPUS_FILE["bytes"]
            and _sha256_file(path) == CORPUS_FILE["sha256"]
        )

    if ok(local):
        return local
    fetched = (downloader or _hub_download)(cache)
    if not ok(fetched):
        size = fetched.stat().st_size if fetched.is_file() else None
        raise ValueError(
            f"{CORPUS_FILE['path']}: fetched {size} bytes, pinned {CORPUS_FILE['bytes']} / "
            f"{CORPUS_FILE['sha256'][:16]}…; refusing to read it"
        )
    return fetched


def read_corpus(path: str | Path) -> list[dict[str, Any]]:
    """The shard's tables as ``{row, chart_type, otsl}`` (text column only; images are read per chosen row by
    `build_sample_dataset`). A row whose OTSL does not parse keeps ``chart_type`` None and is never drawn."""
    import pyarrow.parquet as pq

    texts = pq.read_table(str(path), columns=["texts"]).column("texts").to_pylist()
    out = []
    for index, entry in enumerate(texts):
        if not entry or not isinstance(entry[0], Mapping) or "assistant" not in entry[0]:
            raise ValueError(f"row {index}: no assistant table")
        otsl = str(entry[0]["assistant"])
        try:
            chart_type = parse_otsl(otsl)["chart_type"]
        except ValueError:
            chart_type = None
        out.append({"row": index, "chart_type": chart_type, "otsl": otsl})
    if CORPUS_FILE.get("rows") is not None and len(out) != CORPUS_FILE["rows"]:
        raise ValueError(f"shard has {len(out)} rows, pinned {CORPUS_FILE['rows']}")
    return out


def _image_column(shard_path: str | Path) -> Any:
    import pyarrow.parquet as pq

    return pq.read_table(str(shard_path), columns=["images"]).column("images")


def _image_bytes(column: Any, row: int) -> bytes:
    images = column[row].as_py()
    data = images[0].get("bytes") if images and isinstance(images[0], Mapping) else None
    if not data:
        raise ValueError(f"row {row}: no image bytes")
    return bytes(data)


def _image_suffix(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        fmt = (image.format or "PNG").lower()
    return {"jpeg": ".jpg", "png": ".png", "gif": ".gif", "webp": ".webp"}.get(fmt, ".png")


def stratified_quotas(type_counts: Mapping[str, int], size: int) -> dict[str, int]:
    """Split `size` over the chart types in proportion to `type_counts` (largest remainder, ties by the
    order of CHART_TYPES then name)."""
    total = sum(type_counts.values())
    if total <= 0:
        raise ValueError("no charts to stratify")
    names = sorted(
        type_counts, key=lambda k: (CHART_TYPES.index(k) if k in CHART_TYPES else len(CHART_TYPES), k)
    )
    exact = {k: size * type_counts[k] / total for k in names}
    quotas = {k: int(exact[k]) for k in names}
    for k in sorted(names, key=lambda k: -(exact[k] - quotas[k]))[: size - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def build_sample_dataset(
    rows: Sequence[Mapping[str, Any]],
    shard_path: str | Path,
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
    image_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle each chart type's rows with `seed`, give every split its type quota (`stratified_quotas` of
    the shard's type mix) in the order test, validation, train, and write the chosen chart images to
    `image_dir` under their digest. Charts `validate_dataset` would reject (a target over
    `MAX_TARGET_CHARS`, an image side outside the ceilings) and charts whose image digest was already drawn
    are left out before they are counted, so every split validates and no image is shared."""
    sizes = dict(sizes or SAMPLE_CHARTS)
    out_dir = Path(image_dir) if image_dir is not None else DEFAULT_CACHE_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        if row.get("chart_type") is None:
            continue
        target = target_from_otsl(str(row["otsl"]))
        if len(target) <= MAX_TARGET_CHARS:
            pools.setdefault(str(row["chart_type"]), []).append((int(row["row"]), target))
    type_counts = {name: len(pool) for name, pool in pools.items()}
    quotas = {name: stratified_quotas(type_counts, sizes[name]) for name in ("test", "validation", "train")}
    for name in sorted(pools):
        random.Random(f"{seed}:{name}").shuffle(pools[name])
    out: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    seen: set[str] = set()
    column = _image_column(shard_path)
    for chart_type in sorted(pools):
        candidates = iter(pools[chart_type])
        for split in ("test", "validation", "train"):
            want, taken = quotas[split].get(chart_type, 0), 0
            while taken < want:
                candidate = next(candidates, None)
                if candidate is None:
                    break
                row, target = candidate
                data = _image_bytes(column, row)
                image_id = _sha256_bytes(data)[:16]
                if image_id in seen:
                    continue
                path = out_dir / f"{image_id}{_image_suffix(data)}"
                if not path.is_file() or _sha256_bytes(path.read_bytes())[:16] != image_id:
                    path.write_bytes(data)
                record = {
                    "id": f"{split}-{len(out[split]):04d}",
                    "image_id": image_id,
                    "image": str(path),
                    "target_text": target,
                    "chart_type": chart_type,
                    "row": row,
                }
                try:
                    _check_record(record, 0, base_dir=None)
                except ValueError:
                    continue  # outside the ceilings `validate_dataset` and `extract_table` apply
                seen.add(image_id)
                out[split].append(record)
                taken += 1
    short = {name: (len(out[name]), sizes[name]) for name in out if len(out[name]) < sizes[name]}
    if short:
        raise ValueError(f"the shard's charts do not fill the split targets: {short}")
    return {"train": out["train"], "validation": out["validation"], "test": out["test"]}


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    downloader: Callable[[Path], Path] | None = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned shard."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    shard = fetch_corpus(cache_dir=cache, downloader=downloader)
    return build_sample_dataset(read_corpus(shard), shard, seed=seed, sizes=sizes, image_dir=cache / "images")


# ---- dataset contract ----


def _check_record(record: Any, index: int, *, base_dir: Path | None) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/target_text")
    for key in ("id", "image", "target_text"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    image_ref = record["image"]
    if not isinstance(image_ref, (str, Path)) or not str(image_ref).strip():
        raise ValueError(f"{label}: image must be a file path")
    path = Path(image_ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{label}: image file not found: {path}")
    try:
        with Image.open(path) as handle:
            handle.load()
            validate_image(handle)
            width, height = handle.size
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{label}: image cannot be decoded: {exc}") from exc
    target = record["target_text"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"{label}: target_text must be a non-empty string")
    target = target.strip()
    if len(target) > MAX_TARGET_CHARS:
        raise ValueError(f"{label}: target_text exceeds MAX_TARGET_CHARS={MAX_TARGET_CHARS}")
    table = parse_table(target)
    if not table["rows"]:
        raise ValueError(f"{label}: target_text must hold at least one table row")
    return {
        "id": rid,
        "image_id": str(record.get("image_id", path.name)),
        "image": str(path),
        "image_size": [width, height],
        "target_text": target,
        "table": table,
        "chart_type": str(record.get("chart_type", "other")),
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Structural validation of a chart/table dataset (every image opened and decoded under the ceilings
    `extract_table` applies, every target parsed into at least one row); raises ValueError before any model
    import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, image, target_text} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    base = Path(base_dir) if base_dir is not None else None
    checked = []
    ids: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index, base_dir=base)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    shapes = [(r["table"]["n_rows"], r["table"]["n_columns"]) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_images": len({r["image_id"] for r in checked}),
        "chart_types": dict(Counter(r["chart_type"] for r in checked)),
        "table_rows": {"min": min(s[0] for s in shapes), "max": max(s[0] for s in shapes)},
        "table_columns": {"min": min(s[1] for s in shapes), "max": max(s[1] for s in shapes)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [r["id"], r.get("image_id", ""), str(r["target_text"]).strip(), r.get("chart_type", "")]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no chart image appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record.get("image_id", record["id"]))
            if key in seen and seen[key] != name:
                raise ValueError(f"chart {key!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
    base_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by chart image**: records on the same
    `image_id` land in the same split, so a test chart is never seen in training."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, base_dir=base_dir)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["image_id"], []).append(record)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array or a JSONL file of ``{id, image, target_text}`` objects; `image` paths
    are resolved relative to the file's directory by `validate_dataset(..., base_dir=...)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .json or .jsonl")


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One record per line in the shape `load_byod_dataset` reads back (image paths as given)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ("id", "image_id", "image", "target_text", "chart_type")
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({k: record[k] for k in keys if k in record}, ensure_ascii=False) + "\n")
    return out
