# DePlot chart-to-table extraction pipeline

DIMER inference and fine-tuning wrapper for **DePlot** (`google/deplot`), Google Research's Pix2Struct-based plot-to-table model that translates a chart image into its linearised data table (a `TITLE | …` row, a header row and one row per category, cells separated by `|`, rows by `<0x0A>`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one chart image, renders the fixed DePlot instruction above it with Pillow's bundled font (no Hub font download at inference time), decodes greedily under a caller-owned token budget, and returns the raw linearised text plus a parsed table and a `truncated` flag; it returns no score and answers no questions about the chart. The adaptation contract (`adapt`, `evaluate`, `save_artifact`, `from_artifact`) fine-tunes the decoder's last blocks on chart/table pairs and exports a digest-bound safetensors adapter.

## Upstream alignment

- Model: `google/deplot`
- Revision: `6e76d62430da16986be3426bae32301fb9115397`
- Upstream weight license: Apache-2.0
- Upstream task: plot-to-table translation (chart image → linearised data table), the DePlot modality-conversion module
- Repository adaptation: bounded supervised fine-tuning of the decoder's last *k* blocks plus its final layer norm (`adapt`; the image encoder, embeddings and untied `lm_head` stay frozen); the tutorial's default corpus is one shard of SynthChartNet charts (`docling-project/SynthChartNet` @ `b913ef98a3d45f6136465963ddc71a7a6b0e1728`, `train-00000-of-00135.parquet`, CDLA-Permissive-2.0), downloaded whole at run time and refused unless its size and SHA-256 match the pins recorded by `tools/pin_corpus.py`; its OTSL tables are converted to DePlot's own output convention (bar, pie and stacked-bar tables transposed to one row per category, line tables kept, `TITLE |` first — decided from the frozen model's outputs, see `samples.py`)

## Quick start

```python
from PIL import Image
from deplot_chart_pipeline import DePlotPipeline, cell_accuracy

pipe = DePlotPipeline.from_pretrained()                # stages + verifies weights/deplot first
result = pipe.extract_table(Image.open("chart.png"))   # fixed instruction, greedy, default budget 512
print(result["text"])                                  # 'TITLE | ... <0x0A> Quarter | Revenue <0x0A> Q1 | 120 ...'
print(result["table"]["title"], result["table"]["rows"], result["truncated"])

# relaxed position-wise cell accuracy against a table you know (5 % numeric tolerance)
print(cell_accuracy(result["table"]["rows"], [["Quarter", "Revenue"], ["Q1", "120"], ["Q2", "135"]]))

# bounded fine-tuning on the pinned SynthChartNet sample (downloads the 516 MB shard once)
from deplot_chart_pipeline import fetch_sample_dataset
splits = fetch_sample_dataset()                        # 360 / 80 / 160 charts, stratified by chart type
frozen = pipe.evaluate(splits["test"])                 # cell_accuracy, rnss, exact_table_match, by_chart_type
pipe.adapt(splits["train"], splits["validation"], epochs=3, lr=1e-5, batch_size=4)
pipe.save_artifact("outputs/deplot_adapter")          # adapter.safetensors + manifest.json bound to the base
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `DePlotPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/deplot/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # Pix2StructForConditionalGeneration: 12-layer ViT encoder + 12-layer text decoder
  preprocessor_config.json   # is_vqa: true, max_patches 2048, 16x16 patches
  special_tokens_map.json  spiece.model  tokenizer.json  tokenizer_config.json
  model.safetensors          # git-ignored, 1,129,177,976 bytes
  README.md
```

`pytorch_model.bin` exists upstream and is deliberately not listed (pickle; DIMER does not accept `.bin`).

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`; `INSTRUCTION = "Generate underlying data table of the figure below:"` (fixed; the only prompt DePlot was trained on); `MAX_NEW_TOKENS = 1024`, `DEFAULT_MAX_NEW_TOKENS = 512` (the pinned README example's value); `DECODING = "greedy"`; `MAX_PATCHES = 2048` (the processor's patch budget, documentation only); `ROW_SEPARATOR = "<0x0A>"`, `CELL_SEPARATOR = "|"`; `RELATIVE_TOLERANCE = 0.05` for `cells_match`/`cell_accuracy`. See `MODEL_CARD.md` for who owns the budget, the header-font substitution and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/deplot-chart-pipeline/blob/main/tutorials/deplot_chart_colab.ipynb)

`tutorials/deplot_chart_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the pipeline module, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path resolves the pinned model through the carried staging and verification path, downloads the digest-pinned SynthChartNet shard, converts its tables to the model's own format and splits 360 / 80 / 160 charts by chart type without sharing an image, extracts the table of an 800×520 bar chart drawn in code through `validate_inputs` and `DePlotPipeline.extract_table` (`evaluation_report` `sample-sanity` against the drawn table), scores the frozen model on the test charts (cell accuracy, RNSS, exact-table match, per chart type) beside the empty, header-only and medoid baselines, fine-tunes the decoder's last two blocks (`EPOCHS=3`, `LEARNING_RATE=1e-5`, `BATCH_SIZE=4`) with validation-cell-accuracy epoch selection, scores the held-out charts again, re-extracts the drawn chart, and exports and reloads the adapter with a parity assertion. The held-out gain is recorded as `adapted_beats_frozen`, not asserted. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** for the exact `E2E` carrier recorded in `docs/release-verification.md`: commit `efb92ad` / notebook blob `fb192936d3e6` executed top-to-bottom on Kaggle Tesla T4 on 2026-09-26 UTC (11/11 post-restart code cells, 4381.7 s). On one seeded 160-chart SynthChartNet test split, cell accuracy was 0.160 frozen and 0.181 adapted (medoid baseline 0.037), while RNSS was 0.448 and 0.447 and exact-table match was 0.006 and 0.000; the adapter reloaded with 8/8 identical tables. This qualifies the execution and artifact contract, not a general adaptation gain or a DePlot benchmark. Static/unit checks remain source checks only; any change to the notebook blob returns it to Candidate until a new exact-blob run is recorded.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, header-font note, hosting notes, the SynthChartNet corpus pin and its CDLA-Permissive-2.0 attribution.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; the tutorial's SynthChartNet sample is CDLA-Permissive-2.0 (attribution to SynthChartNet, docling-project); see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
