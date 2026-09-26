# Release verification

`tutorials/deplot_chart_colab.ipynb` (`E2E`, **standalone** carrier) is **Candidate** at source revision `ddf0cc4`
and notebook blob `90d9093cced4`. It returns to **Release-grade** only after that exact blob executes top-to-bottom
in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 2.0 (REL8). This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  SynthChartNet dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `DePlotPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` and `read_corpus` from the pinned cache path,
  `target_from_otsl` on one raw table, `build_sample_dataset(corpus_rows, shard_path, seed=SPLIT_SEED, image_dir=...)`
  / `load_byod_dataset`, `validate_dataset` per split, `check_split_disjoint`, `write_dataset_jsonl`, the ceiling
  print, `validate_inputs` with the zero-budget refusal probe, `pipe.extract_table` with the sanity checks,
  `evaluation_report` on the drawn chart, `empty_baseline`, `header_only_baseline`, `medoid_baseline`,
  `pipe.evaluate` on the frozen model and on the validation and test splits after adaptation with the per-chart-type
  breakdown, `pipe.adapt` with its explicit hyperparameters, `evaluation_report` on the drawn chart after adaptation,
  `pipe.save_artifact`, `DePlotPipeline.from_artifact` and the reload-parity assertion, the recorded
  `frozen_beats_empty` and `adapted_beats_frozen` flags, and the provenance fields `weight_format`, `weight_sha256`
  and the `corpus` block), the six expected `outputs/` paths, the learner-facing statements and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `Pix2StructForConditionalGeneration` /
  `Pix2StructProcessor` / `.generate(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` /
  `safetensors` / `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline suites (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner and shard downloader, tiny PIL
drawings, a small generated parquet shard with OTSL tables of all four chart types, temporary manifests, no weights).
`tests/test_adaptation_model.py` builds a 3-layer, 32-wide random Pix2Struct from the committed config, VQA processor
and tokenizer (header rendered in Pillow's bundled font) and runs the real `evaluate` / `adapt` / `save_artifact` /
`load_artifact` path on it offline, including a tampered-adapter refusal; its pinned-checkpoint and CUDA cases skip
unless `model.safetensors` is staged and a GPU is visible. These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; a GPU runtime is recommended for Sections 6–8) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-runtime evidence for the recorded blob |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/deplot/` or the corpus cache `weights/synthchartnet/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `TABLE_MAX_TOKENS = 512`, `EPOCHS = 3`, `LEARNING_RATE = 1e-5`,
   `BATCH_SIZE = 4`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; an interpreter restart after the install is expected where the runtime's preinstalled
   torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `google/deplot` at the immutable revision
     on a clean runtime, `verify_snapshot` returning its dict (8 files), and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loading from the verified directory with no further Hub access (a font fetch in the logs after staging is a
     finding — the header font is Pillow's bundled one);
   - Section 4: `fetch_corpus` downloading `train-00000-of-00135.parquet` at the pinned dataset revision and
     accepting it only with the pinned size (515,825,444 bytes) and SHA-256; `read_corpus` returning 14,568 rows
     (bar 6,491 / pie 6,665 / stacked_bar 1,333 / line 79); one stacked-bar OTSL table printed beside its transposed
     target; the chart-type-stratified draw of 360 / 80 / 160 training, validation and test charts with
     `check_split_disjoint` reporting no shared image (the local pin run realised train bar 161 / pie 165 /
     stacked_bar 32 / line 2, validation 36 / 37 / 7 / 0, test 72 / 73 / 14 / 1); the three dataset digests;
     `outputs/deplot_chart_train.jsonl` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings surfaced; the bar chart drawn; `validate_inputs` writing
     `outputs/deplot_chart_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     zero-budget probe); `pipe.extract_table` with every sanity check `True` and `evaluation_report` verdict
     `sample-sanity` (the inference-only notebook's runs read 9 of 10 cells and the title; a different table on
     another runtime is a finding to record, not a failure);
   - Section 6: the empty, header-only and medoid baselines and the frozen model's test cell accuracy, RNSS,
     exact-table match, truncation rate and per-chart-type scores, and `frozen_beats_empty` / `frozen_beats_medoid`
     printed;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 18,879,744 trainable of 282,285,696 parameters
     (29 tensors: two decoder blocks and the final layer norm), the training-chart count, and the epoch history with
     validation cell accuracy and RNSS;
   - Section 8: `pipe.evaluate` on the validation and test splits with the five-way comparison, the per-chart-type
     breakdown, `adapted_beats_frozen` printed and `outputs/deplot_chart_evaluation_report.json` written (the gain is
     **recorded, not asserted**, until a measured recipe is on file);
   - Section 9: the drawn chart extracted by the adapted model with the `sample-sanity` report,
     `outputs/deplot_chart_table.csv` written; `pipe.save_artifact` writing
     `outputs/deplot_chart_adapter/{adapter.safetensors,manifest.json}` and `DePlotPipeline.from_artifact` reloading
     it with identical tables on eight test charts (the cell asserts it); `outputs/deplot_chart_result.json` written
     with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block, the `corpus` block (with the target
     rule), the inference-contract items, the comparison, the artifact digest, the reload parity, the runtime versions
     and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark), the value of
   `frozen_beats_empty` and `adapted_beats_frozen` and any warning or applicable `SHOULD` deviation in the tables
   below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Recorded executions

Notebook identity is the Git blob id of `tutorials/deplot_chart_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/deplot_chart_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### `E2E` notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-26 (04:24–04:27) | `60de8e1` / `d46a685b9e19` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-deplot-chart` v2), serial suite; exact committed blob fetched and Git-blob verified; clean Hugging Face cache | Default sample path, all sections | 213.6 s | **FAILED** — after the expected install-cell restart, 4/11 code cells completed; Section 4 raised `TypeError: _hub_download() takes 1 positional argument but 2 were given` because the standalone carrier placed `samples.py` and `pipeline.py` in one namespace and both defined `_hub_download`. The helper was renamed to `_download_corpus`, a namespace regression test was added, and the notebook was regenerated. Not promotion evidence. |
| 2026-09-26 (04:49–06:02) | `efb92ad` / `fb192936d3e6` (`NOTEBOOK_SOURCE.repository_revision` = `aada162`, the source revision the notebook was generated at; `aada162..efb92ad` changes only the notebook) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-deplot-chart` v3), serial suite; exact committed blob fetched and Git-blob verified, executed verbatim in a fresh interpreter with a `google.colab` shim and no repository checkout; Hugging Face cache clean at start; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`, Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after inline pins: torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, device `cuda:0`, float32 | Default sample path (`USE_BYOD = False`, all form defaults), all sections. The 8-file DePlot snapshot and pinned SynthChartNet shard were fetched into an empty cache and digest-verified; 621 staged files / 1,670,135,452 bytes. Corpus split 360 / 80 / 160 charts; test chart types bar 72, line 1, pie 73, stacked-bar 14. Frozen test: cell accuracy 0.160, RNSS 0.448, exact-table match 0.006, truncation 0.050; empty/header-only/medoid cell-accuracy baselines 0.000 / 0.007 / 0.037. Adaptation trained 18,879,744 of 282,285,696 parameters for three epochs, best epoch 2 by validation cell accuracy (0.123 frozen → 0.162), 1999.5 s. Adapted test: cell accuracy 0.181 (+0.021), RNSS 0.447 (−0.001), exact-table match 0.000; `adapted_beats_frozen` is true because that flag uses cell accuracy, not because every metric improved. Drawn-chart cell accuracy remained 0.9. Adapter 29 tensors / 75,522,608 bytes, fresh reload parity 8/8 identical tables. Preserved SHA-256: result `1050078a1a17…`, evaluation report `b627f6d1f2ec…`, input manifest `3f74fd5ca99d…`, table CSV `be3928cd4bdf…`, training JSONL `0fca6f83b5ef…`, adapter `c04d63c37554…`, adapter manifest `b39f75c1cfd5…` | 4381.7 s (pass 1 198.3 s stopped at the install cell with the expected stale-module guard; kernel restarted; pass 2 4183.4 s) | **PASSED** — 11/11 post-restart code cells; promotion evidence for this exact blob. One seeded split of one shard on one runtime, no dispersion estimate; not a DePlot benchmark. |
| — | — | Local harness (pre-flight) | Default sample path, all sections | — | not yet executed |

Local builder evidence that is **not** a notebook execution: the frozen checkpoint was run on CPU on twelve charts of
the pinned shard (4 pie, 4 bar, 2 stacked-bar, 2 line) to decide the target format — every output began with
`TITLE |` (empty title), pies had no header row, stacked bars came out with the legend series as columns and the axis
categories as rows, line charts with the x points as rows — and `tools/pin_corpus.py` confirmed the shard pin
(SHA-256 `1d6cf578…b209e7`, 14,568 rows) and realised the split above. The pinned-checkpoint cases of
`tests/test_adaptation_model.py` (evaluate, a one-epoch adaptation with artifact round trip, the final-epoch reload,
the tensor-set and tampered-digest refusals, the transactional guarantee) passed on CPU against the staged snapshot.

### Superseded `TASK-INFERENCE` notebook — local pre-flight (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `3381d03cf223` (commit `df0e1ed`, generated at `7f0202d`; `NOTEBOOK_SOURCE.repository_revision` = `7f0202d…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (1.13 GB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font download in the log, `extract_table` → 56 tokens, `truncated` false, title exact, rows `Quarter | Quarterly revenue`, `Q1 | 120`, `Q2 | 135`, `Q3 | 150`, `Q4 | 180`, `evaluation_report` `sample-sanity` (`cell_accuracy` 0.9 = 9/10, `title_match` 1.0), 5 outputs written | 118.3 s | PASS — pre-flight only; not promotion evidence |

### Superseded `TASK-INFERENCE` notebook — manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `1a74479` / `ead762313bf6` | Kaggle CPU (`kurtvalcorza/dimer-nb2-deplot-chart` v1) | Default sample path | 288.1 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 1133 MB staged |

## Current status

**Candidate** for source revision `ddf0cc4` / notebook blob `90d9093cced4`. The carried split contract now groups
records by both declared image identity and SHA-256 of the image bytes, and adaptation rollback covers epoch-0
validation and progress callbacks. Those source changes invalidate the prior exact-blob qualification. The passing
`efb92ad` / `fb192936d3e6` Kaggle Tesla T4 run above remains historical evidence only; a clean run of the current blob
is required before promotion.

Facts a reviewer should weigh before promotion: SynthChartNet is not part of DePlot's fine-tuning mixture, so this is
adaptation to a new chart family; the target format is a decision recorded in `samples.py` from the frozen model's own
outputs (bar, pie and stacked-bar OTSL tables transposed, line tables kept, `TITLE |` first), and cell accuracy is
position-wise, so part of the measured gain can be layout learning rather than better reading — RNSS, which ignores
layout, did not improve; the fine-tuning recipe (`LEARNING_RATE = 1e-5`, three epochs, two blocks, batch 4) has one
measurement and the notebook records `adapted_beats_frozen` rather than asserting a general gain; line charts are
0.5 % of the shard, so the stratified split
holds one or two per split and none in validation, and per-type line numbers are anecdotes; SynthChartNet carries some
label noise (for example a pie whose first category cell is a unit caption); the shard is 516 MB, a large download for
a tutorial; the header uses Pillow's bundled font rather than the Arial the checkpoint was trained with; each chart
costs an encoder pass at up to 2,048 patches plus autoregressive decoding of up to `TABLE_MAX_TOKENS`, in evaluation
as in training, so a GPU runtime is recommended; and the 80-chart validation and 160-chart test splits carry no
dispersion estimate.
