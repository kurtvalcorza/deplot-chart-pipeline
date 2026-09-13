"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "deplot_chart_pipeline",
    "repo_name": "deplot-chart-pipeline",
    "stem": "deplot_chart",
    "notebook_name": "deplot_chart_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "DePlotPipeline",
    "weights_key": "deplot",
    "runtime_imports": ["torch", "transformers"],
    "title": "DePlot — DIMER chart-to-table extraction tutorial (standalone)",
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
    "capability": "chart-to-table extraction — one chart image → its linearised data table (title, header row, data rows) — using the pinned `google/deplot` weights",
    "intro": (
        "At inference the Pix2Struct image-encoder/text-decoder (a ViT-style encoder over variable-resolution 16×16 patches "
        "and a 12-layer text decoder, 282M parameters, pretrained by parsing masked web screenshots into HTML and fine-tuned "
        "by Google Research on plot-to-table data as DePlot) reads the fixed instruction **rendered as a text header above "
        "the chart** — the Pix2Struct convention — scales the composite to fill at most 2048 patches, and generates a "
        "linearised table: rows separated by the literal token `<0x0A>`, cells by `|`, with a leading `TITLE | …` row when a "
        "title was read. Decoding is greedy (`do_sample=False`) under a caller-owned `max_new_tokens` budget. **No "
        "adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this "
        "notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the carried module adds "
        "snapshot verification, the input contract (image side ceilings, the token budget), a fixed output contract, an "
        "offline header font (Pillow's bundled Aileron replaces the Hub font the upstream processor would otherwise "
        "download), `parse_table`, and the `cell_accuracy`, `validate_inputs` and `evaluation_report` helpers. The default "
        "sample is a bar chart drawn in code from a known data table, so relaxed cell accuracy and the title match are "
        "demonstration (plumbing) evidence for one chart, not a chart-to-table benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic bar chart from a known table (or upload your own chart) and "
        "validate it into an input manifest, choose a token budget, run the supported task, read the linearised table "
        "correctly (separators, title row, parsed rows, the `truncated` flag, no score), exercise an optional BYOD path, "
        "produce an evaluation report that is `sample-sanity` with relaxed `cell_accuracy` and `title_match` only when the "
        "expected table exists and `not-measurable` otherwise, and export the table as JSON and CSV with provenance."
    ),
    "exclusions": (
        "chart question answering or reasoning over the extracted table (the DePlot paper pairs the table with an LLM; none "
        "is bundled), charts in images with several plots or a plot embedded in a page (one chart image per call), any "
        "instruction other than the fixed plot-to-table prompt, batch throughput, sampling or beam search, evaluation on "
        "ChartQA/PlotQA or the paper's relative-mapping-similarity metric (only position-wise relaxed cell accuracy against "
        "a table you supply is computed here), and any training. The model generates a table for any image, including one "
        "with no chart, and gives no signal when it invents."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.4 s to load and 6.3 s for the 800×520 drawn bar chart (56 generated tokens) in the Windows venv (Intel Core Ultra 9 275HX); cost scales with the tokens generated. The pinned `torch==2.14.0` install and the 1.13 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; how a chart's data table is laid out (header row, one row per category); that a well-formed table is not a correct one.",
        "- **Data:** the default sample is a deterministic 800×520 bar chart drawn in code with Pillow's bundled font — a title, a labelled y axis with gridlines, four quarterly bars with x labels and value labels — from a five-row data table you can read in the code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) of a **single chart** (bar, line or pie, ideally with axis labels and a title), any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic bar chart or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference: a bar chart — the title `Quarterly revenue "
                "2025 (USD millions)`, a y axis from 0 to 200 with gridlines and tick labels, four bars for Q1–Q4 with their "
                "values printed above them and an x-axis caption — is drawn with Pillow's bundled font at 800×520 from the "
                "five-row table `EXPECTED_TABLE` (header row first), the same chart the repository's smoke run used. That table "
                "and the title are the references for the `cell_accuracy` and `title_match` sanity checks later; they are not a "
                "labelled dataset, so nothing here is a benchmark, and a drawn chart with printed values is far easier than a "
                "typical published chart. The image digest is printed for the record. BYOD is optional and disabled by default; "
                "when enabled, upload one chart image — no expected table exists for it, so the evaluation report will be "
                "`not-measurable`.\n\n"
                "The token budget is a **caller-owned request parameter**: `max_new_tokens` bounds the linearised table "
                "(`DEFAULT_MAX_NEW_TOKENS = 512`, the pinned README example's value; `MAX_NEW_TOKENS = 1024` is the ceiling). "
                "The instruction is fixed — DePlot was trained on exactly `INSTRUCTION` and the pipeline exposes no other. "
                "Nothing is validated in this cell — the next section hands the image and the budget to the pipeline's own "
                "validation stage, which is the only checker. Look for a dictionary naming the sample kind, the image size and "
                "digest, the budget and the expected table's shape."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "max_new_tokens = 512  # @param {{type:\"integer\"}}\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    expected_table, expected_title = None, None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic chart: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image = bar_chart()\n"
                "    expected_table, expected_title = EXPECTED_TABLE, CHART_TITLE\n"
                "    image_name = 'synthetic_bar_chart_800x520.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'max_new_tokens': max_new_tokens, 'expected_shape': None if expected_table is None else [len(expected_table), len(expected_table[0])]}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `extract_table` "
                "applies — image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px and `max_new_tokens` in "
                "`[1, MAX_NEW_TOKENS]` — and returns an **input manifest** naming the schema (including the fixed instruction, "
                "the header-rendering preprocessing, the output conventions and the decoding rule), the input's observed mode "
                "and size, the budget and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show "
                "what rejection looks like, the cell also validates a zero budget and records the pipeline's own error message "
                "as a finding. Inside the pipeline the image is converted to RGB, the instruction is rendered above it, and the "
                "composite is scaled to the patch budget; nothing else is dropped or altered. The pipeline cannot tell whether "
                "the image is a chart: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'INSTRUCTION': INSTRUCTION, 'ROW_SEPARATOR': ROW_SEPARATOR, 'CELL_SEPARATOR': CELL_SEPARATOR}}}})\n"
                "input_manifest = validate_inputs(image, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, max_new_tokens=0)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'zero-budget-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Extract the table and read the output correctly\n\n"
                "`extract_table` returns a dict with `text` — the linearised table exactly as decoded — `table` (the parsed "
                "form from `parse_table`: `title` or `None`, `rows` of stripped cells, `n_rows`, `n_columns`), the fixed "
                "`instruction`, `image_size`, `new_tokens`, a `truncated` flag that is true when the budget was exhausted, the "
                "generation settings and the model identity. **No score exists**: the table is generated text with no "
                "probability and no correctness signal, and a well-formed table is not evidence that its numbers were read from "
                "the chart. Greedy decoding is deterministic on a fixed device and dtype; CUDA kernel selection can change a "
                "token and therefore the rest of the table, so GPU and CPU outputs need not match. As recorded in the model card, "
                "the repository's CPU smoke on this same chart generated 56 tokens in 6.3 s: the title exactly, all four "
                "quarters and values exactly, and the header `Quarter | Quarterly revenue` — the y axis has no label, so the "
                "model filled the second header cell from the title, the one cell the reference calls `Revenue`. That is one "
                "observation on a drawn chart with printed values, not a calibration point."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.extract_table(image, max_new_tokens=max_new_tokens)\n"
                "elapsed = time.time() - t0\n"
                "table = result['table']\n"
                "print({{'seconds': round(elapsed, 1), 'new_tokens': result['new_tokens'], 'truncated': result['truncated'], 'device': pipe.device, 'title': table['title'], 'shape': [table['n_rows'], table['n_columns']]}})\n"
                "print(result['text'])\n"
                "for row in table['rows']:\n"
                "    print(' | '.join(f'{{cell:>14}}' for cell in row))\n"
                "if result['truncated']:\n"
                "    print('The token budget was exhausted: the table is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No chart-to-table "
                "metric is reported by default: the DePlot paper's relative mapping similarity and ChartQA-style accuracy need "
                "chart images paired with their data tables, and this repository ships none. The repository's metric helper is "
                "`cell_accuracy` — position-wise relaxed matching of every expected cell against the predicted cell at the same "
                "position, text cells compared case/whitespace-insensitively and numeric cells within 5 % relative tolerance "
                "(the ChartQA relaxed-accuracy convention), reported with both tables' shapes — plus a `title_match` entry when "
                "an expected title is supplied; with an expected table the verdict is `sample-sanity`. On the synthetic path "
                "that table is data **you rendered yourself** with the values printed on the bars, so a high accuracy proves "
                "only that the input contract, header rendering, forward pass, decoding and parsing round-trip. On BYOD no "
                "expected table exists, the verdict is `not-measurable`, and the report states what would make the task "
                "measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, expected_table, expected_title=expected_title, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'metrics'}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    if metric['id'] == 'cell_accuracy':\n"
                "        print(f\"cell_accuracy {{metric['value']:.3f}}  ({{metric['matched']}}/{{metric['total']}} cells; predicted shape {{metric['predicted_shape']}}, expected {{metric['expected_shape']}})\")\n"
                "    else:\n"
                "        print(f\"title_match   {{metric['value']:.0f}}  (predicted {{metric['predicted']!r}}, expected {{metric['expected']!r}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No expected table exists for this input, so nothing is scored; compare the rows with the chart yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (linearised text, parsed table, `new_tokens`, `truncated`, the "
                "budget), the evaluation report, the input manifest, the sample identity, digest, expected table and title, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, "
                "device). The parsed table is also written as CSV (one line per row, the title as a leading comment line when "
                "present), and a side-by-side PNG places the chart above a panel with the extracted rows for visual inspection — "
                "a supplement to, not a replacement for, the machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "panel_lines = [' | '.join(row) for row in table['rows']][:16] or ['(no rows)']\n"
                "panel_height = 24 + 22 * len(panel_lines)\n"
                "annotated = Image.new('RGB', (image.width, image.height + panel_height), 'white')\n"
                "annotated.paste(image.convert('RGB'), (0, 0))\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "draw.line([(0, image.height + 1), (image.width, image.height + 1)], fill=(120, 120, 120), width=2)\n"
                "panel_font = ImageFont.load_default(size=16)\n"
                "for index, line in enumerate(panel_lines):\n"
                "    draw.text((16, image.height + 10 + 22 * index), line[:120], fill=(40, 90, 220), font=panel_font)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "with open('outputs/{stem}_table.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    if table['title']:\n"
                "        handle.write(f\"# title: {{table['title']}}\\n\")\n"
                "    csv.writer(handle).writerows(table['rows'])\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'expected_table': expected_table, 'expected_title': expected_title}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The table is what the model generates after reading the chart with the instruction printed above it; nothing in the "
        "output scores it, and a tidy table can carry misread values, invented headers or missing rows. On the synthetic chart "
        "the `cell_accuracy` and `title_match` values in the evaluation report compare the output with a table you rendered "
        "yourself — with every value printed on its bar — and the verdict is `sample-sanity`, which proves only that the input "
        "contract, header rendering, forward pass, decoding and parsing work (the repository's smoke run matched 9 of 10 cells, "
        "the miss being the unlabelled y-axis header); they say nothing about charts without value labels, stacked or grouped "
        "bars, line charts with many points, pie charts, log axes, legends, unusual styles or non-English labels, and a BYOD "
        "result is a single-chart observation with the verdict `not-measurable`. **The model generates a table for any "
        "image**: a blank image yielded `TITLE | <0x0A> | % <0x0A> Eli | 55.1 <0x0A> Sarah | 44.9` in the smoke run — an "
        "invented two-row table with no signal — so an image that is not a chart produces a confident fabrication rather "
        "than an empty result, and `truncated` is the only structural flag you get. The pipeline provides no chart QA, no "
        "reasoning, no multi-chart pages, no alternative instructions and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** remove the value labels from `bar_chart` (delete the `d.text(... value ...)` line) and see how "
        "many cells survive when the model must read bar heights against the axis; add a y-axis label `Revenue` and check "
        "whether the header cell is fixed; lower `max_new_tokens` to 20 and watch `truncated` turn true; enable `USE_BYOD` "
        "with a published chart whose data you know, then pass the table as `expected_table` to `evaluation_report` to see "
        "the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- DePlot: One-shot visual language reasoning by plot-to-table translation (Liu et al., 2022): https://arxiv.org/abs/2212.10505\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., 2022): https://arxiv.org/abs/2210.03347\n"
        "- ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning (Masry et al., 2022): https://arxiv.org/abs/2203.10244"
    ),
}
