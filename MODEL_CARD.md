---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: visual-question-answering
task: "Others - Chart Understanding"
base_model: google/deplot
date_published: "2023-04-03"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2023-04-03T11:05:38Z, https://huggingface.co/api/models/google/deplot — the Transformers-format conversion); the DePlot paper is arXiv:2212.10505 (2022-12) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# DePlot — Chart-to-Table Extraction (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fdeplot-ffcc4d?style=flat)](https://huggingface.co/google/deplot)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/pix2struct)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2212.10505-b31b1b.svg)](https://arxiv.org/abs/2212.10505)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/deplot-chart-pipeline/blob/main/tutorials/deplot_chart_colab.ipynb) [`deplot_chart_colab.ipynb`](https://github.com/kurtvalcorza/deplot-chart-pipeline/blob/main/tutorials/deplot_chart_colab.ipynb)  
  *Table extraction from a bar chart drawn in code with the pinned `google/deplot` weights: a linearised, parsed data table under a caller-owned token budget, and relaxed `cell_accuracy` / `title_match` against the drawn table as sanity evidence only — no ChartQA benchmark.*

---

#### Description

`google/deplot` is the Transformers-format release of DePlot, the plot-to-table modality-conversion module from "DePlot: One-shot visual language reasoning by plot-to-table translation" (Liu et al., arXiv:2212.10505; Google Research), converted from the T5X checkpoints by the Hugging Face team and pinned here to revision `6e76d62430da16986be3426bae32301fb9115397` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `Pix2StructForConditionalGeneration`: the Pix2Struct *base* architecture — a 12-layer ViT-style image encoder (hidden size 768, 12 heads, 16×16 patches with learned row/column position embeddings over a variable-resolution input of up to 4096 patch positions) and a 12-layer T5-style text decoder (vocabulary 50,244) — 282M parameters in the 1.13 GB float32 `model.safetensors`, pretrained by parsing masked web screenshots into HTML and fine-tuned on the paper's standardised plot-to-table corpus (synthetic and real charts paired with their data tables). At inference the VQA processor (`Pix2StructImageProcessor`, `preprocessor_config.json`: `is_vqa` true, `max_patches` 2048) **renders the fixed instruction `Generate underlying data table of the figure below:` as a text header above the chart**, scales the composite to fill at most 2048 patches while preserving aspect ratio, normalises it per image and flattens it into patch tokens; the decoder then generates the linearised table — rows separated by the literal token `<0x0A>`, cells by `|`, with a leading `TITLE | …` row when a title was read. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `DePlotPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`, refusing a snapshot whose processor is not the VQA variant), `header_font_bytes` (Pillow's bundled Aileron Regular passed to the image processor so the upstream inference-time download of `ybelkada/fonts/Arial.TTF` never happens — a deliberate, documented departure from the upstream rendering font), `extract_table` (input validation, the fixed instruction, greedy decoding under a caller-owned `max_new_tokens`, a `truncated` flag), `parse_table`, `cells_match`, `cell_accuracy`, and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is plot-to-table translation: input one chart image (`PIL.Image.Image`, any mode, converted to RGB) and a token budget; output the linearised table text, its parsed form (title, rows, shape), the number of tokens generated and whether the budget was exhausted. Envisioned applications are the first stage of the DePlot recipe — turning a bar, line or pie chart into a table that a human or a language model then reasons over — for chart digitisation in reports and papers, accessibility (a textual rendering of a chart), and indexing of figure data, with the extracted numbers checked against the chart before use. Within DIMER the pipeline is an inference component and a zero-configuration baseline for chart digitisation, not a certified extractor for any chart style, renderer or domain.

###### Primary Intended Users

Intended users are machine-learning engineers, document-processing developers, and data analysts integrating chart digitisation into research prototypes or internal document tooling. A user is expected to understand that the output is *generated text* — it carries no per-cell confidence and no correctness signal, and the model produces a table for any image, including one with no chart — that numbers read from bar heights or line positions are estimates whose precision depends on the chart's axis and labels (the tutorial prints every value on its bar, which real charts rarely do), that headers can be filled from titles or invented when the axes are unlabelled (the smoke run's one miss), that the token budget is theirs to set (a table with many rows can exceed 512 tokens and be reported `truncated`), that the instruction is fixed and rendered into the image with a font this repository supplies (not the one used at fine-tuning time), that the training data is chart renders in English so hand-drawn charts, screenshots with clutter, non-English labels and unusual chart types are distribution shifts, that greedy decoding is reproducible on a fixed device but GPU and CPU outputs need not match, and that accuracy can only be measured on charts paired with their tables that they supply. Users who need chart question answering, reasoning, multi-chart pages, other instructions or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no chart question answering or numerical reasoning (the paper pairs DePlot's table with a large language model; none is bundled), no instruction other than the fixed plot-to-table prompt, no multi-chart pages or charts embedded in documents (one chart image per call), no confidence per cell, no batching, no sampling or beam search, and no abstention (the model cannot say "not a chart").
2. **Input boundary:** `extract_table` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, and budgets outside `[1, MAX_NEW_TOKENS = 1024]` (`ValueError`/`TypeError`). Every chart is scaled to fill at most 2048 patches of 16×16 px, so axis ticks and labels that are only a few pixels tall at that scale are unlikely to be read, and the rendered header consumes part of the budget. The bundled header font covers printable ASCII; the fixed instruction uses nothing else.
3. **Input boundary:** the fine-tuning data is the paper's standardised plot-to-table corpus — synthetic charts and charts from ChartQA/PlotQA-style sources, rendered, in English, with conventional bar/line/pie styles. Hand-drawn or photographed charts, screenshots with surrounding UI, 3-D or stacked/grouped charts with legends, log axes, dual axes, dense line charts, infographics and non-English labels fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. An image that is not a chart at all still produces a table (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on extracted numbers — financial reporting, scientific meta-analysis, regulatory or clinical data extraction — without a human comparing the table with the chart, and a locally measured cell accuracy on the deployment's own labelled chart set.

#### Factors

###### Groups

This pipeline is not human-centric by design: it transcribes a chart's data table and never classifies, identifies or scores people. The fine-tuning data (per the paper: synthetic charts plus ChartQA- and PlotQA-derived charts with their tables) contains no evaluation groups in the demographic sense, and neither the upstream authors nor this repository audited it for anything of the kind. What does vary is the chart population: the corpus is English-language, conventionally styled, machine-rendered charts, so charts with non-English or non-Latin labels, unusual or culturally specific styles, accessibility-oriented palettes, hand-drawn or historical charts and infographic layouts are the groups whose extraction accuracy is unknown, not known to be equal. Where charts encode data about people — survey results by demographic group, health statistics, employment figures — errors in the extracted numbers propagate into whatever analysis follows; the operator who digitises such charts is responsible for validating the tables against the sources before drawing conclusions about groups.

###### Instrumentation

The upstream training "instrument" is a chart renderer (matplotlib-style synthetic charts and the renderers behind ChartQA/PlotQA sources) paired with the exact table used to draw each chart, with the instruction rendered above the image in the Arial font Pix2Struct preprocessing uses. Inference images arrive from whatever produced them — a PDF figure at some DPI, a screenshot, a scan of a printed report, a phone photograph of a slide — and resolution, compression, anti-aliasing, colour and clutter all change the visual evidence; the 2048-patch budget fixes the encoder's input size regardless of source, and a small chart in a large page loses its tick labels. This repository renders the instruction with Pillow's bundled Aileron rather than Arial, a second instrument change whose effect was measured only on the synthetic chart (title and all values exact). The pipeline validates type and size only; it cannot detect a low-DPI figure, a cropped axis, a legend it will mis-assign, or an image that is not a chart. The synthetic tutorial chart (Pillow's bundled font, printed value labels, one series, a titled axis frame) is itself a rendering instrument cleaner and easier than most published charts.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (float32) but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 8-file, 1.13 GB snapshot 0.58 s; load 5.44 s; one 800×520 drawn bar chart 6.3 s for 56 generated tokens; a blank 800×520 image with a 64-token budget 4.1 s for 27 tokens — cost is dominated by the encoder pass over the 2048-patch input plus autoregressive decoding, so it grows with the table's length. Data environment: the model assumes a single, conventionally styled, machine-rendered chart with readable axes; the synthetic chart satisfies that assumption (and adds printed value labels) and is where the measured behaviour holds. Photographs, cluttered screenshots, unusual chart types and non-English labels violate it to degrees this repository did not measure, and the pipeline reports no signal when they do — nor when the image is not a chart at all.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. The table is generated text with no score, probability or correctness signal; `new_tokens` and `truncated` describe the generation, and `parse_table` reports the shape — descriptions of the output, not metrics. The repository ships `cells_match` (equal after case/whitespace normalisation, or numerically within `RELATIVE_TOLERANCE = 0.05` — the ChartQA relaxed-accuracy convention, after stripping `$`, `%` and thousands separators) and `cell_accuracy` (position-wise: every expected cell scored against the predicted cell at the same row and column, so a missing row or column is a miss; both shapes reported), because they are the primitives a caller would use to evaluate; the paper's relative mapping similarity (RMS), which aligns rows and columns before scoring, is not implemented. Both need charts paired with their tables that the caller must supply; ChartQA and PlotQA are not bundled. The public `evaluation_report(result, expected_rows=None, *, expected_title=None)` stage returns that report in machine-readable form: a `cell_accuracy` entry (matched, total, both shapes) and, when an expected title is given, a `title_match` entry, with the verdict `sample-sanity`; or the verdict `not-measurable` naming the labelled set that would be required when no expected table is supplied. The paper's ChartQA and PlotQA numbers (for DePlot + an LLM) are upstream-reported and this pipeline does not reproduce or claim them.

###### Decision thresholds

No score threshold exists: the model generates tokens until end-of-sequence or the budget and nothing is filtered. The decision parameter is the **token budget** `max_new_tokens`, default `DEFAULT_MAX_NEW_TOKENS = 512` (the value the pinned README's example passes; the synthetic chart needed 56) with ceiling `MAX_NEW_TOKENS = 1024`. A budget that is too small is reported, not hidden: `truncated` is true whenever `new_tokens` reaches it, and the caller should raise the budget and rerun. The instruction is fixed by the model's training and is not a parameter. Decoding is greedy (`do_sample=False`) with no temperature, beams or repetition penalty. The evaluation helper carries one threshold of its own, `RELATIVE_TOLERANCE = 0.05`, the relaxed-accuracy convention of the ChartQA literature, not a value tuned here; a deployment that needs exact numbers should set it to 0. A deployment owns choosing the budget per chart family and deciding how an extracted table is verified against the chart before it is used.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, token counts and the extracted table on one synthetic chart, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but because decoding is autoregressive a single differing token changes the rest of the table, and CPU and CUDA outputs need not match; the synthetic chart's own bytes depend on the Pillow build's bundled font. On the synthetic chart the model read the title exactly and all eight category/value cells exactly, and filled the unlabelled y-axis header with `Quarterly revenue` where the reference says `Revenue` (`cell_accuracy` 0.9, `title_match` 1.0) — one observation on one easy chart with printed values, not an estimate; on a blank image it produced an invented two-row table (`Eli | 55.1`, `Sarah | 44.9`), which is what an unconditioned generator with no abstention looks like. A caller who needs an accuracy estimate must supply labelled charts and compute cell accuracy or RMS over many charts or bootstrap resamples themselves; a caller who needs a confidence per cell has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The paper describes pretraining on 80M masked web-page screenshots paired with simplified HTML (Pix2Struct) and fine-tuning on a standardised plot-to-table corpus built from synthetic charts and from ChartQA and PlotQA sources (charts scraped from statistics sites and generated from public data tables); web screenshots can contain personal data, and statistical charts can encode data about people, so personal data in the training corpus is not ruled out and was not audited here. This repository distributes code, tests, and documentation; it does not distribute the 1,129,177,976-byte `model.safetensors`, which is staged locally under `weights/deplot/` and git-ignored, and it ships no sample charts — the tutorial chart is drawn in code from fictitious numbers. The operator must audit the charts they submit for proprietary or otherwise restricted content; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — digitising clinical-trial or epidemiological charts for evidence synthesis, financial charts for investment or audit decisions, safety or engineering plots for compliance — would be admissible only with human comparison of every extracted cell against the chart (the model estimates numbers from pixels and invents headers and rows without signal), a locally measured cell accuracy on the deployment's own labelled chart set, a documented budget and verification policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 8 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, refuses a snapshot whose image processor is not the VQA variant, and the smoke run loaded and extracted with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) is neither listed nor loaded. **The inference-time font download the upstream processor performs (`ybelkada/fonts/Arial.TTF`, unpinned, unlisted) is replaced** by `header_font_bytes()` — Pillow's bundled Aileron Regular (CC0) — and the loader calls the image processor directly because `Pix2StructProcessor.__call__` drops the `font_bytes` keyword. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported; another asserts the header font bytes are TrueType and stable.
- **Input integrity:** the public `validate_inputs(image, *, max_new_tokens)` stage applies exactly the checks `extract_table` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the fixed instruction, the budget and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; out-of-range or boolean budgets are rejected; `extract_table` raises on a malformed runner result; `cell_accuracy` rejects an empty expected table.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; greedy decoding with no sampling; a fixed instruction and a fixed, bundled header font; the output conventions are module constants; every result carries `model_id`, `model_revision`, the instruction, the budget, `new_tokens`, `truncated`, the device and the dtype.
- **Refusals:** no batching, no download without the explicit flag, no Hub access at inference time, no alternative prompts, no sampling, no pickle deserialisation, no attempt to guess whether the image is a chart.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Invented tables:** the model has no abstention — a blank image yielded a two-row table with names and percentages in the smoke run — so a non-chart image, a cropped figure or a decorative graphic produces a plausible table with no signal; downstream analyses that trust it are built on fabrication.
- **Estimated numbers and invented headers:** values read from bar heights or line positions are pixel estimates, and an unlabelled axis is filled from context (the smoke run's `Quarterly revenue` header); a table that looks complete can be quietly wrong, and only cell accuracy on labelled charts reveals the rate.
- **Truncation and loops:** a long table can exceed the budget (reported via `truncated`) and a repetition loop runs to the budget; a caller who ignores the flag ships a partial table.
- **Automation bias:** a tidy table invites trust that generated text has not earned; the header-font substitution adds an untested variable.
- **Misattribution in complex charts:** legends, grouped or stacked bars and dual axes can be assigned to the wrong series without any signal — none of these was exercised.
- **Bias amplification:** any chart style, language or domain the English render-heavy corpus under-represents is reproduced as uneven accuracy, undetected because no per-family evaluation exists.
- **Resource use:** a 1.13 GB model and ~6 s per chart on the reference CPU; a figure stream saturates a shared host, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: digitising charts in order to misrepresent their data, to fabricate or launder evidence, or to present extracted numbers as verified source data; processing charts the operator has no right to process, or paywalled and licence-restricted publications in breach of their terms; deceptive uses in which extracted tables are passed downstream without disclosure that they were machine-estimated; and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the material processed. Autonomous high-consequence actions triggered by unreviewed extracted tables are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `google/deplot`
- Revision: `6e76d62430da16986be3426bae32301fb9115397`
- Snapshot manifest: `weights/deplot/dimer-base-manifest.json`, 8 files, `totalBytes` 1133308796
- `model.safetensors` SHA-256: `ab90055611f42fee327d9ecf3c9cdac63e847bd19a0ac8ea86b0e8134fe0711b` (1,129,177,976 bytes, float32)
- `config.json` SHA-256: `f1dc8bcbda2ac4f8715c2d5003d3afb591ac0af97782fdf95a65cf26f805c5bb` (4,883 bytes; `Pix2StructForConditionalGeneration`)
- `preprocessor_config.json` SHA-256: `c84e4eebc84171d6069533d9f0147ec7b4afd02ab78697cb5c30f9419ef7dc45` (249 bytes; `is_vqa` true, `max_patches` 2048, 16×16 patches)
- Weight format: SafeTensors; loader `Pix2StructForConditionalGeneration.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `Pix2StructProcessor` from the same directory, the image processor called with `header_text=INSTRUCTION` and `font_bytes=header_font_bytes()`. The upstream `pytorch_model.bin` is not part of the manifest and is never loaded.

## Input/output contract

- `DePlotPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `extract_table(image, *, max_new_tokens=512) -> dict` with keys `text` (linearised table, stripped), `table` (`{"title": str | None, "rows": [[str, ...], ...], "n_rows": int, "n_columns": int}`), `instruction`, `image_size`, `new_tokens`, `truncated`, `generation` (`max_new_tokens`, `do_sample` false, `decoding` greedy), `device`, `dtype`, `source`, `model_id`, `model_revision`.
- `parse_table(text) -> dict`; `cells_match(predicted, expected, *, relative_tolerance=0.05) -> bool`; `cell_accuracy(predicted_rows, expected_rows, *, relative_tolerance=0.05) -> dict`; `header_font_bytes() -> bytes`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `INSTRUCTION`, `MAX_NEW_TOKENS = 1024`, `DEFAULT_MAX_NEW_TOKENS = 512`, `DECODING = "greedy"`, `MAX_PATCHES = 2048`, `ROW_SEPARATOR = "<0x0A>"`, `CELL_SEPARATOR = "|"`, `RELATIVE_TOLERANCE = 0.05`, `INPUT_SCHEMA`.
- `validate_inputs(image, *, max_new_tokens, names) -> dict`; `evaluation_report(result, expected_rows=None, *, expected_title=None, sample_kind) -> dict`; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing renders the fixed instruction as a header (Pillow's bundled Aileron, 36 pt, black on white) above the chart and scales the composite to at most 2048 16×16 patches (`Pix2StructImageProcessor`, snapshot defaults); greedy decoding.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.58 s (8 files, 1.13 GB); load 5.44 s; `extract_table` on a synthetic 800×520 bar chart (title `Quarterly revenue 2025 (USD millions)`, a 0–200 y axis with gridlines and ticks, four bars Q1–Q4 with values 120/135/150/180 printed above them, drawn with Pillow's bundled font) at `max_new_tokens=512` → 56 new tokens, `truncated` false, 6.3 s; output `TITLE | Quarterly revenue 2025 (USD millions) <0x0A> Quarter | Quarterly revenue <0x0A> Q1 | 120 <0x0A> Q2 | 135 <0x0A> Q3 | 150 <0x0A> Q4 | 180`; `evaluation_report` against the drawn table: `cell_accuracy` 0.9 (9/10; the y-axis header `Revenue` came back as `Quarterly revenue`), `title_match` 1.0, verdict `sample-sanity`; 800×520 blank image with `max_new_tokens=64` → `TITLE |  <0x0A>  | % <0x0A> Eli | 55.1 <0x0A> Sarah | 44.9` (27 tokens, 4.1 s).
- Tutorial execution: `tutorials/deplot_chart_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 118.3 s including the 1.13 GB staging, same table as the smoke run); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, charts without printed value labels, line or pie charts, legends or grouped series, any accuracy measurement against a labelled chart set, photographs or screenshots of charts, non-English labels, the upstream Arial header rendering for comparison.

## References

- Liu et al. DePlot: One-shot visual language reasoning by plot-to-table translation. ACL Findings 2023. https://arxiv.org/abs/2212.10505
- Lee et al. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. ICML 2023. https://arxiv.org/abs/2210.03347
- Masry et al. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. ACL Findings 2022. https://arxiv.org/abs/2203.10244
- Methani et al. PlotQA: Reasoning over Scientific Plots. WACV 2020. https://arxiv.org/abs/1909.00997
- Upstream code: https://github.com/google-research/pix2struct
- Upstream card: https://huggingface.co/google/deplot
- Transformers `Pix2Struct` documentation: https://huggingface.co/docs/transformers/model_doc/pix2struct
- Aileron font (Pillow's bundled default, CC0): https://dotcolon.net/fonts/aileron
