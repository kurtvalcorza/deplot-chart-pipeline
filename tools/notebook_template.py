"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E chart-to-table workflow: the pinned google/deplot snapshot is digest-verified
and loaded, one digest-pinned parquet shard of SynthChartNet charts is downloaded, converted to DePlot's own
table format, validated and split by chart type, a drawn bar chart is extracted through the inference contract,
the frozen model is scored on the held-out charts beside three non-neural baselines, a bounded fine-tuning of
the decoder's last blocks runs in the kernel, the held-out split is scored again per chart type, the adapted
model re-extracts the drawn chart, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "deplot_chart_pipeline",
    "repo_name": "deplot-chart-pipeline",
    "stem": "deplot_chart",
    "notebook_name": "deplot_chart_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `google/deplot` snapshot (a 1.13 GB `model.safetensors`), downloads one digest-pinned parquet shard of "
        "SynthChartNet charts from the Hugging Face Hub (516 MB, no credential, refused on any size or SHA-256 mismatch), "
        "converts each chart's OTSL table into DePlot's own linearised format, draws a seeded chart-type-stratified sample of "
        "360 training, 80 validation and 160 test charts with no image shared between splits, extracts the table of a drawn "
        "bar chart through the inference contract with an input manifest and a rejection probe, scores the frozen model on "
        "the test charts beside three non-neural baselines, runs a bounded fine-tuning of the decoder's last blocks with "
        "validation-cell-accuracy epoch selection, scores the held-out charts again per chart type, re-extracts the drawn "
        "chart with the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a "
        "fresh pipeline to verify table parity. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). A CUDA runtime is used automatically "
        "when present; the CPU path works but is slow (every chart is encoded at up to 2,048 patches and its table generated "
        "token by token), and the timings of the first clean run are recorded in `docs/release-verification.md`."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `records.jsonl` (or `records.json`) of objects with `id`, `image` (a file name inside the zip), "
        "`target_text` (the chart's table in DePlot's linearised format: `TITLE | <title>` then rows joined by ` <0x0A> ` and "
        "cells by ` | `) and optional `image_id` and `chart_type`, beside the image files. They pass through the same "
        "validation, seeded image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and "
        "reload-parity cells as the SynthChartNet sample. The expected schema and the ceilings are stated in the "
        "Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the "
        "default path."
    ),
    "pipeline_class": "DePlotPipeline",
    "weights_key": "deplot",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "DePlot — DIMER E2E chart-to-table fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/deplot-chart-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/deplot-chart-pipeline/blob/main/tutorials/deplot_chart_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fdeplot-ffcc4d?style=flat",
            "https://huggingface.co/google/deplot",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/pix2struct",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2212.10505-b31b1b.svg", "https://arxiv.org/abs/2212.10505"),
    ],
    "capability": "chart-to-table extraction and bounded supervised fine-tuning of the decoder's last blocks on a chart/table dataset, using the pinned `google/deplot` weights",
    "intro": (
        "`google/deplot` is the DePlot model of Liu et al. (2022) — a Pix2Struct image encoder over variable-resolution 16×16 "
        "patches (up to 2,048 per image) and a 12-layer text decoder, 282,285,696 parameters, fine-tuned by Google Research "
        "to translate a plot into its underlying data table — published under the **Apache-2.0** licence. The fixed "
        "instruction is **rendered as a text header above the chart** (the Pix2Struct convention, in Pillow's bundled font so "
        "no font is downloaded), the composite is encoded, and the decoder generates a linearised table with greedy decoding "
        "under a caller-owned `max_new_tokens` budget: a `TITLE | …` row, then rows joined by the literal token `<0x0A>` and "
        "cells by `|`. **No score exists**: the table is generated text with no probability and no correctness signal, and a "
        "well-formed table is **not evidence that its numbers** were read from the chart.\n\n"
        "What this notebook adds to inference is **adaptation with labelled tables**. The dataset is real chart/table pairs "
        "from a chart family DePlot was not fine-tuned on: SynthChartNet (docling-project), bar, pie, stacked-bar and line "
        "charts rendered with Matplotlib, Seaborn and Pyecharts from financial-report tables, published under the "
        "**CDLA-Permissive-2.0** licence. The notebook downloads **one pinned parquet shard** (516 MB, SHA-256 pinned in the "
        "carried module) and draws a seeded, chart-type-stratified subset from it. **Target format:** SynthChartNet stores "
        "each table as OTSL, with the axis categories of a bar, pie or stacked-bar chart along its first row; the frozen "
        "checkpoint, run on charts of this shard, writes one row per category instead (and a line chart's x points as rows, "
        "as OTSL already does), so the carried converter transposes bar, pie and stacked-bar tables, keeps line tables, and "
        "writes them in the model's own `TITLE |` / ` <0x0A> ` / ` | ` convention. The honest question is narrow: does a "
        "bounded adaptation of the decoder's last blocks on a few hundred charts of a new family move held-out cell accuracy "
        "at all, and on which chart type?\n\n"
        "Scores are **relaxed position-wise cell accuracy** (the repository's metric: every expected cell compared with the "
        "predicted cell at the same position, numbers within 5 %), **RNSS** (the relative number set similarity of the "
        "ChartQA and DePlot papers, which ignores layout) and exact-table match, overall and per chart type. Three references "
        "frame them: the **empty baseline**, the **header-only baseline** and the **medoid baseline** — systems that never "
        "look at the chart. Nothing here is a quality claim about your charts: it is one seeded split of one shard.\n\n"
        "**Weight-format note:** the pinned revision ships the model as SafeTensors (`model.safetensors`, digest-pinned in "
        "the manifest, loaded in float32); the processor is the VQA variant, which renders the header. Section 3 stages and "
        "digest-verifies the snapshot before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; download a digest-pinned shard of chart/table pairs, convert its "
        "tables to the model's own output convention, validate them and split them by chart type without sharing an image; "
        "extract a table through the public API on a drawn chart and read `text`, the parsed `table`, `new_tokens` and "
        "`truncated` correctly (generated text, no score); score the frozen model's cell accuracy and RNSS beside three "
        "non-neural baselines and read the per-chart-type breakdown; run a bounded fine-tuning with explicit hyperparameters "
        "and validation-based epoch selection; evaluate on a held-out test split; re-extract a drawing from a different "
        "image family with the adapted model; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "chart question answering or reasoning over the extracted table (the DePlot paper pairs the table with a language "
        "model; none is bundled), charts in images with several plots or a plot embedded in a page (one chart image per "
        "call), any instruction other than the fixed plot-to-table prompt, batch throughput, sampling or beam search, the "
        "DePlot paper's relative mapping similarity (RMS) and evaluation on ChartQA or PlotQA, fine-tuning of the image "
        "encoder, the embeddings or the output projection, training on charts that are not the pinned sample or your own "
        "uploads, and any claim that a SynthChartNet split stands in for your charts. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available; a GPU runtime is recommended for Sections 6–8. Every chart is encoded at up to 2,048 patches and its table generated token by token, so each extraction costs seconds on CPU. The pinned `torch==2.14.0` install, the 1.13 GB checkpoint and the 516 MB chart shard are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; how a chart's data table is laid out (one row per category, one column per series); that a well-formed table is not a correct one; what a score against baselines that never see the image does and does not show.",
        "- **Data contract:** records carry `id`, `image` and `target_text` — an image file decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px, and the chart's table as a DePlot linearised string of at most `MAX_TARGET_CHARS` (512) characters with at least one row; optional `image_id` groups related records on the same chart and optional `chart_type` labels the breakdown. Validation also hashes every image, so byte-identical files remain in one split even under different names or ids. Ids match `[A-Za-z0-9_.:-]` (1–64 characters) and are unique; a dataset needs 8..5,000 records; every record on the same chart lands in the same split. BYOD accepts one zip of images plus a `records.jsonl` / `records.json` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every target parsed, but nothing checks that a table is the chart's real data — a mislabelled chart is fine-tuned on without complaint, and SynthChartNet itself carries some label noise (for example a pie whose first category cell is a unit caption).",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path downloads one object from the Hub dataset repository `docling-project/SynthChartNet` at the immutable revision `b913ef98…` (`train-00000-of-00135.parquet`, 515,825,444 bytes) and refuses it unless its size and SHA-256 match the pins carried in `samples.py`; the chart images are written to the cache under their own content digest. SynthChartNet is published under CDLA-Permissive-2.0, which permits use and places no restriction on results such as trained weights; attribution to SynthChartNet (docling-project) is required when the data is shared.",
    ],
    "cells": [
        {
            "md": (
                "## 4. SynthChartNet charts, tables and split\n\n"
                "`fetch_corpus` returns the pinned shard from the cache under `weights/synthchartnet/` or downloads it at the "
                "pinned dataset revision, and refuses it unless its byte size and SHA-256 equal the pins in the carried module "
                "(it also refuses to run at all while no SHA-256 pin is recorded). `read_corpus` reads the table column with "
                "`pyarrow` and parses every OTSL table. `build_sample_dataset` converts each table to its target with "
                "`target_from_otsl` — bar, pie and stacked-bar tables transposed to one row per category, line tables kept, "
                "`<ecel>` as an empty cell, `TITLE | ` first — leaves out the few charts whose target exceeds "
                "`MAX_TARGET_CHARS`, shuffles each chart type with `SPLIT_SEED`, and fills the test, validation and training "
                "splits to `SAMPLE_CHARTS` in proportion to the shard's type mix (so line charts, 0.5 % of the shard, get one "
                "or two charts per split, or none). A chart whose image digest was already drawn is skipped, so no image is "
                "shared between splits. `validate_dataset` then opens every image and parses every target, "
                "`check_split_disjoint` asserts no image is shared, and the training split is written to "
                "`outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: the shard's row count and type mix, one raw OTSL table beside its converted target, the chart "
                "counts per split and per type, the table shapes, three digests, and four refusal probes — a duplicate id, a "
                "missing image file, a target with no table row and a dataset too small to split — each rejected before "
                "`torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import io\n"
                "import json\n"
                "import zipfile\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
                "        for member in archive.infolist():\n"
                "            name = Path(member.filename).name\n"
                "            if member.is_dir() or not name or name.startswith('.'):\n"
                "                continue\n"
                "            (byod_dir / name).write_bytes(archive.read(member))\n"
                "    records_file = next(p for p in (byod_dir / 'records.jsonl', byod_dir / 'records.json') if p.is_file())\n"
                "    records = load_byod_dataset(records_file)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED, base_dir=byod_dir)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    shard_path = fetch_corpus(cache_dir='weights/synthchartnet')\n"
                "    corpus_rows = read_corpus(shard_path)\n"
                "    raw_rows = {{'charts': len(corpus_rows), 'chart_types': dict(collections.Counter(r['chart_type'] for r in corpus_rows))}}\n"
                "    example_row = next(r for r in corpus_rows if r['chart_type'] == 'stacked_bar')\n"
                "    print({{'otsl': example_row['otsl'][:220], 'target': target_from_otsl(example_row['otsl'])[:220]}})\n"
                "    splits = build_sample_dataset(corpus_rows, shard_path, seed=SPLIT_SEED, image_dir='weights/synthchartnet/images')\n"
                "    data_source = f'{{CORPUS_NAME}} — {{CORPUS_RELEASE}}'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "chart_types = {{name: manifest['chart_types'] for name, manifest in dataset_manifests.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'shard_sha256': str(CORPUS_FILE['sha256'])[:16] + '...'}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'charts': manifest['n_records'], 'images': manifest['unique_images'], 'chart_types': manifest['chart_types'], 'table_rows': manifest['table_rows'], 'table_columns': manifest['table_columns'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'chart_type': example['chart_type'], 'target_text': example['target_text'][:200]}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.png'}}, *splits['train'][1:8]],\n"
                "    'target without a table row': [{{**splits['train'][0], 'target_text': 'TITLE | nothing else'}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Extract a table through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a bar chart drawn in code at "
                "800×520 from the five-row table `EXPECTED_TABLE` — the title `Quarterly revenue 2025 (USD millions)`, a y axis "
                "from 0 to 200 with gridlines, four bars for Q1–Q4 with their values printed above them. It is a different "
                "image family from the SynthChartNet charts, and the adapted model will extract it again in Section 9. "
                "`validate_inputs` applies exactly the checks `extract_table` applies (image sides "
                "`MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, `max_new_tokens` in `[1, MAX_NEW_TOKENS]`) and returns an input manifest; a "
                "zero budget is validated too and its rejection recorded as a finding. `extract_table` returns `text` (the "
                "linearised table exactly as decoded), the parsed `table`, `new_tokens`, a `truncated` flag that is true when "
                "the budget was exhausted, and the model identity. **No score exists.** As recorded in the model card, the "
                "repository's CPU smoke on this chart generated 56 tokens: the title and all four quarters and values exactly, "
                "and the header `Quarter | Quarterly revenue` where the reference says `Revenue`. `evaluation_report` scores it "
                "against the table you drew yourself — verdict `sample-sanity`, plumbing evidence, not a metric. The image "
                "digest depends on the Pillow build's bundled font rendering."
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "TABLE_MAX_TOKENS = 512  # @param {{type:\"integer\"}}\n"
                "CHART_TITLE = 'Quarterly revenue 2025 (USD millions)'\n"
                "EXPECTED_TABLE = [['Quarter', 'Revenue'], ['Q1', '120'], ['Q2', '135'], ['Q3', '150'], ['Q4', '180']]\n\n\n"
                "def bar_chart(width=800, height=520):\n"
                "    \"\"\"Titled bar chart with a labelled y axis, gridlines, x labels and value labels, drawn from EXPECTED_TABLE.\"\"\"\n"
                "    img = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(img)\n"
                "    title_f, tick_f, label_f = ImageFont.load_default(size=24), ImageFont.load_default(size=16), ImageFont.load_default(size=18)\n"
                "    d.text((width / 2 - d.textlength(CHART_TITLE, font=title_f) / 2, 24), CHART_TITLE, fill='black', font=title_f)\n"
                "    x0, y0, x1, y1 = 100, 80, width - 40, height - 80\n"
                "    d.line([(x0, y0), (x0, y1), (x1, y1)], fill='black', width=2)\n"
                "    ymax = 200\n"
                "    for v in range(0, ymax + 1, 50):\n"
                "        y = y1 - (y1 - y0) * v / ymax\n"
                "        d.line([(x0, y), (x1, y)], fill=(200, 200, 200), width=1)\n"
                "        d.text((x0 - 12 - d.textlength(str(v), font=tick_f), y - 9), str(v), fill='black', font=tick_f)\n"
                "    n = len(EXPECTED_TABLE) - 1\n"
                "    slot = (x1 - x0) / n\n"
                "    for i, (label, value) in enumerate(EXPECTED_TABLE[1:]):\n"
                "        bx0 = x0 + slot * i + slot * 0.25\n"
                "        bx1 = bx0 + slot * 0.5\n"
                "        by = y1 - (y1 - y0) * int(value) / ymax\n"
                "        d.rectangle([bx0, by, bx1, y1], fill=(60, 110, 200), outline=(20, 50, 120))\n"
                "        d.text(((bx0 + bx1) / 2 - d.textlength(label, font=label_f) / 2, y1 + 12), label, fill='black', font=label_f)\n"
                "        d.text(((bx0 + bx1) / 2 - d.textlength(value, font=tick_f) / 2, by - 22), value, fill='black', font=tick_f)\n"
                "    d.text((width / 2 - 40, height - 40), 'Quarter', fill='black', font=label_f)\n"
                "    return img\n\n\n"
                "image = bar_chart()\n"
                "image_name = 'synthetic_bar_chart_800x520.png'\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_TARGET_TOKENS': MAX_TARGET_TOKENS, 'MAX_TARGET_CHARS': MAX_TARGET_CHARS, 'DECODING': DECODING, 'INSTRUCTION': INSTRUCTION, 'ROW_SEPARATOR': ROW_SEPARATOR, 'CELL_SEPARATOR': CELL_SEPARATOR, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(image, max_new_tokens=TABLE_MAX_TOKENS, names=[image_name])\n"
                "try:\n"
                "    validate_inputs(image, max_new_tokens=0)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'zero-budget-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'image': image_name, 'rgb_sha256': image_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "started = time.perf_counter()\n"
                "result = pipe.extract_table(image, max_new_tokens=TABLE_MAX_TOKENS)\n"
                "result_seconds = round(time.perf_counter() - started, 3)\n"
                "print(result['text'])\n"
                "checks = {{\n"
                "    'text_is_str': isinstance(result['text'], str),\n"
                "    'budget_respected': result['new_tokens'] <= TABLE_MAX_TOKENS,\n"
                "    'setting_echoed': result['generation']['max_new_tokens'] == TABLE_MAX_TOKENS and result['generation']['do_sample'] is False,\n"
                "    'not_truncated': not result['truncated'],\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'extract_table output failed a sanity check: {{checks}}')\n"
                "frozen_drawn = evaluation_report(result, EXPECTED_TABLE, expected_title=CHART_TITLE, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': result_seconds, 'new_tokens': result['new_tokens'], 'frozen_drawn_verdict': frozen_drawn['verdict'], 'cell_accuracy': frozen_drawn['metrics'][0]['value'], 'title_match': frozen_drawn['metrics'][1]['value']}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test charts\n\n"
                "Three references frame the adaptation, and none looks at the chart. The **empty baseline** emits no table "
                "(the floor: cell accuracy 0). The **header-only baseline** emits, per chart type, the most frequent header "
                "row of the training targets and no data rows — what a system that knows only the layout convention scores. "
                "The **medoid baseline** emits, per chart type, the training table that best matches the other training "
                "tables of its type. The **frozen model** extracts every test chart with the budget from Section 5 and is "
                "scored by `pipe.evaluate`: mean relaxed **cell accuracy** (position-wise, so a missing header row or a "
                "transposed table shifts every cell), **RNSS** (numbers only, layout-free), exact-table match and the "
                "truncation rate, overall and per chart type. SynthChartNet is not DePlot's training family and its targets "
                "follow the converter's layout rule, so expect the frozen model's cell accuracy well below its numbers on its "
                "own benchmarks; whether it beats the empty table is recorded as `frozen_beats_empty` rather than assumed. The "
                "measured values of the first clean run are recorded in `docs/release-verification.md` and the model card."
            ),
            "code": (
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_header = header_only_baseline(train_records, test_records)\n"
                "baseline_medoid = medoid_baseline(train_records, test_records)\n"
                "for baseline in (baseline_empty, baseline_header, baseline_medoid):\n"
                "    print({{'baseline': baseline['baseline'], 'cell_accuracy': round(baseline['cell_accuracy'], 3), 'rnss': round(baseline['rnss'], 3), 'note': baseline['note']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=TABLE_MAX_TOKENS)\n"
                "print({{'frozen_model_test': {{'cell_accuracy': round(frozen_test['cell_accuracy'], 3), 'rnss': round(frozen_test['rnss'], 3), 'exact_table_match': round(frozen_test['exact_table_match'], 3), 'truncated_rate': round(frozen_test['truncated_rate'], 3)}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'by_chart_type': {{'medoid': baseline_medoid['by_chart_type'], 'frozen': frozen_test['by_chart_type']}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for row in frozen_test['rows'][:3]:\n"
                "    print({{'chart_type': row['chart_type'], 'target': row['target_text'][:160], 'frozen': row['prediction'][:160], 'cell_accuracy': round(row['cell_accuracy'], 3)}})\n"
                "frozen_beats_empty = frozen_test['cell_accuracy'] > baseline_empty['cell_accuracy']\n"
                "frozen_beats_medoid = frozen_test['cell_accuracy'] > baseline_medoid['cell_accuracy']\n"
                "print({{'frozen_beats_empty': frozen_beats_empty, 'frozen_beats_medoid': frozen_beats_medoid}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the decoder's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` blocks of the text decoder plus the decoder's "
                "final layer norm — two blocks by default, 18,879,744 of 282,285,696 parameters; the image encoder, every "
                "embedding and the untied output projection stay frozen. Each training chart is one sample: the fixed "
                "instruction is rendered above the chart exactly as `extract_table` renders it, the frozen encoder reads the "
                "composite (recomputed each step without gradients), and the target is the tokenised linearised table "
                "(at most `MAX_TARGET_TOKENS` tokens) with its end-of-sequence token, decoded with teacher forcing and scored "
                "with the model's own cross-entropy (padding ignored); AdamW at a fixed learning rate, gradient clipping at "
                "1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is "
                "scored on the validation charts and the epoch with the highest validation cell accuracy is kept (ties keep "
                "the earlier one). If no epoch beats the frozen model on validation, the selector keeps epoch 0 and the adapter "
                "reproduces the frozen tables; that outcome is reported, not hidden."
            ),
            "code": (
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_cell_accuracy': round(entry['val']['cell_accuracy'], 3), 'val_rnss': round(entry['val']['rnss'], 3)}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_charts': adapt_result['n_train'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test charts were never used for training or epoch selection, and no test image appears in the training or "
                "validation splits. The adapted model is scored exactly as the frozen model was in Section 6, and the five "
                "systems — the three baselines, the frozen and the adapted model — are put side by side overall and per chart "
                "type. Read it in this order: **cell accuracy** first (the metric the epoch was selected on), then **RNSS** "
                "(an adaptation that only learns the target layout raises cell accuracy without reading the numbers any better; "
                "RNSS ignores layout), then the per-type split — stacked-bar and line charts are a small share of the test "
                "split. `adapted_beats_frozen` records whether held-out cell accuracy rose. A test split of 160 charts from one "
                "seeded draw of one shard gives **no dispersion estimate**, so the deltas are sample-sanity evidence that the "
                "adaptation contract works, not a benchmark, and a gain on SynthChartNet says nothing about your charts until "
                "you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=TABLE_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=TABLE_MAX_TOKENS)\n"
                "comparison = {{\n"
                "    'cell_accuracy': {{'empty': round(baseline_empty['cell_accuracy'], 3), 'header_only': round(baseline_header['cell_accuracy'], 3), 'medoid': round(baseline_medoid['cell_accuracy'], 3), 'frozen': round(frozen_test['cell_accuracy'], 3), 'adapted': round(adapted_test['cell_accuracy'], 3)}},\n"
                "    'rnss': {{'empty': round(baseline_empty['rnss'], 3), 'header_only': round(baseline_header['rnss'], 3), 'medoid': round(baseline_medoid['rnss'], 3), 'frozen': round(frozen_test['rnss'], 3), 'adapted': round(adapted_test['rnss'], 3)}},\n"
                "    'exact_table_match': {{'frozen': round(frozen_test['exact_table_match'], 3), 'adapted': round(adapted_test['exact_table_match'], 3)}},\n"
                "    'delta_vs_frozen': {{'cell_accuracy': round(adapted_test['cell_accuracy'] - frozen_test['cell_accuracy'], 3), 'rnss': round(adapted_test['rnss'] - frozen_test['rnss'], 3)}},\n"
                "    'by_chart_type': {{chart_type: {{'n': row['n'], 'medoid': baseline_medoid['by_chart_type'][chart_type]['cell_accuracy'], 'frozen': row['cell_accuracy'], 'adapted': adapted_test['by_chart_type'][chart_type]['cell_accuracy']}} for chart_type, row in frozen_test['by_chart_type'].items()}},\n"
                "}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "for before, after in list(zip(frozen_test['rows'], adapted_test['rows'], strict=True))[:3]:\n"
                "    print({{'target': before['target_text'][:160], 'frozen': before['prediction'][:160], 'adapted': after['prediction'][:160]}})\n"
                "adapted_beats_frozen = adapted_test['cell_accuracy'] > frozen_test['cell_accuracy']\n"
                "print({{'adapted_beats_frozen': adapted_beats_frozen}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'chart_types': chart_types,\n"
                "    'max_new_tokens': TABLE_MAX_TOKENS,\n"
                "    'baselines': {{'empty': baseline_empty, 'header_only': baseline_header, 'medoid': baseline_medoid}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'frozen_beats_empty': frozen_beats_empty,\n"
                "    'frozen_beats_medoid': frozen_beats_medoid,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-extract the drawn chart, export the adapter and reload it\n\n"
                "The drawn bar chart from Section 5 is extracted again by the adapted model and scored against the table you "
                "drew — a different image family from the SynthChartNet charts it was tuned on, and one with a title and a "
                "header the SynthChartNet targets never carry, so this is a small look at whether the adaptation changed the "
                "model's behaviour *outside* its sample (one chart of evidence, not a measurement; a different table here is a "
                "finding to record, not a failure). Both tables are written as CSV.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the decoder's last two blocks and its final layer norm, "
                "about 76 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id "
                "and revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `DePlotPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses "
                "any tensor that is not a decoder tensor, and overlays the tensors onto a freshly loaded base — a new object "
                "from files, not the in-memory model (VER2). The cell asserts identical tables on eight test charts (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_result = pipe.extract_table(image, max_new_tokens=TABLE_MAX_TOKENS)\n"
                "adapted_drawn = evaluation_report(adapted_result, EXPECTED_TABLE, expected_title=CHART_TITLE, sample_kind='synthetic')\n"
                "print({{'frozen': result['text'], 'adapted': adapted_result['text']}})\n"
                "print({{'drawn_chart_cell_accuracy': {{'frozen': frozen_drawn['metrics'][0]['value'], 'adapted': adapted_drawn['metrics'][0]['value']}}}})\n"
                "with open('outputs/{stem}_table.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['model', 'row', 'cells'])\n"
                "    for label, extracted in (('frozen', result), ('adapted', adapted_result)):\n"
                "        for index, row in enumerate(extracted['table']['rows']):\n"
                "            writer.writerow([label, index, ' | '.join(row)])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = DePlotPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [r['text'] for r in pipe.predict(test_records[:8], max_new_tokens=TABLE_MAX_TOKENS)]\n"
                "after = [r['text'] for r in reloaded.predict(test_records[:8], max_new_tokens=TABLE_MAX_TOKENS)]\n"
                "parity = {{'identical_tables': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_tables'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'SafeTensors, loaded in float32, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'file': CORPUS_FILE, 'sample_charts': SAMPLE_CHARTS, 'target_rule': 'bar, pie, stacked_bar transposed; line kept; TITLE row first'}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'drawn_chart': {{'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'expected_table': EXPECTED_TABLE, 'expected_title': CHART_TITLE}}, 'frozen': {{k: result[k] for k in ('text', 'new_tokens', 'truncated')}}, 'frozen_report': frozen_drawn, 'adapted': {{k: adapted_result[k] for k in ('text', 'new_tokens', 'truncated')}}, 'adapted_report': adapted_drawn, 'seconds': result_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'frozen_beats_empty': frozen_beats_empty,\n"
                "    'frozen_beats_medoid': frozen_beats_medoid,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is a plot-to-table translator scored on charts of a family it was not fine-tuned on, beside three "
        "baselines that never look at the chart, and a bounded fine-tuning of the decoder's last two blocks on a few hundred "
        "charts is then scored on a held-out test split — overall and per chart type — with an adapter that reloads to "
        "identical tables. That is the claim: the adaptation contract works end to end on a real labelled chart corpus, and "
        "the numbers it produces are read against the baselines and the frozen model rather than in isolation. Whether "
        "held-out cell accuracy rose is recorded as `adapted_beats_frozen`, not assumed.\n\n"
        "Cell accuracy is position-wise, so part of any gain can be the model learning the converter's layout — no header row "
        "for a single unnamed series, the transposed stacked bars — rather than reading values better; RNSS, which ignores "
        "layout, is the check on that. The test split is 160 charts from one seeded draw of one shard, the validation split "
        "that picks the epoch is 80, stacked-bar and line charts are a small share of both, and SynthChartNet carries some "
        "label noise. **The model generates a table for any image**: an image that is not a chart produces a confident "
        "fabrication rather than an empty result, and `truncated` is the only structural flag you get. Fine-tuning on a "
        "narrow sample can also erode the model elsewhere; the drawn chart re-extracted in Section 9 is one chart of evidence "
        "about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty, header-only and medoid tables on *your* charts "
        "are the numbers to read before any model's, per chart type. **Layout is part of the target:** decide your table "
        "orientation and header rule once, from what the frozen model emits, and convert every label to it — a transposed "
        "target scores as a miss on every cell. **Leakage:** keep every record of a chart in one split (the contract does "
        "this) and split by source document when your charts come from few reports.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled chart "
        "corpus, validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning, evaluate against three trivial baselines and the frozen model on a held-out split, and emit the shown "
        "machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, accuracy on any other chart population, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `LEARNING_RATE` and watch the training loss "
        "fall while the validation cell accuracy drops and the selector keeps an early epoch; set `TRAINABLE_DECODER_LAYERS = "
        "1` and compare the artifact size and the held-out scores; remove the value labels from `bar_chart` and see how many "
        "cells survive; or bring your own charts through BYOD and read the baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Google, Apache-2.0): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- DePlot: One-shot visual language reasoning by plot-to-table translation (Liu et al., 2022): https://arxiv.org/abs/2212.10505\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., ICML 2023): https://arxiv.org/abs/2210.03347\n"
        "- ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning (Masry et al., 2022): https://arxiv.org/abs/2203.10244\n"
        "- SynthChartNet (docling-project, CDLA-Permissive-2.0): https://huggingface.co/datasets/docling-project/SynthChartNet\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
