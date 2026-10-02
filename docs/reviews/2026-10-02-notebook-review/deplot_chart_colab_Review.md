# DePlot Chart-to-Table E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/deplot-chart-pipeline`  
**Notebook:** `tutorials/deplot_chart_colab.ipynb`  
**Reviewed commit:** `232c8e1b199f9f2b46904e757f03f19cc6eefc5c` (`main`, confirmed with `gh api repos/kurtvalcorza/deplot-chart-pipeline/commits/main`)  
**Notebook Git blob:** `90d9093cced431645aa7680f620d3095da23b101`. This is the blob of the recorded Kaggle Tesla T4 v4 run of 2026-09-26 (commit `dd5724e`); `git diff --stat dd5724e 232c8e1` touches only `MODEL_CARD.md`, `README.md`, `STATUS.md`, `docs/release-verification.md` and `tutorials/README.md`. `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `DPC`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` (`b1cfe13`).

## Executive assessment

This is a careful, honest E2E notebook. It carries the three package modules verbatim (their metadata digests equal the repository files; the generator check passes), stages and re-hashes the pinned 8-file `google/deplot` snapshot, downloads one digest-pinned SynthChartNet shard, converts its OTSL tables to the model's own output convention, draws an image-disjoint, chart-type-stratified 360/80/160 split, runs the inference contract on a drawn chart with an input manifest and a refusal probe, scores the frozen model beside three non-neural baselines, fine-tunes two decoder blocks with validation-based epoch selection, evaluates on the untouched test split overall and per chart type, and exports a safetensors adapter that reloads to identical tables. The prose is unusually explicit about what the numbers do not show: no score, position-wise cell accuracy versus layout-free RNSS, one seeded split, label noise, a gain that may be layout learning.

What this review observed directly (CPU, install skipped, model staged locally; Section 4's 516 MB shard not downloaded):

| Measure | This review | Kaggle T4 record (blob `90d9093c`) |
|---|---|---|
| Setup cells 3, 5, 7, 9, 11 | all ok; 8/8 files verified, `fetched: []` (staged) | ok after a restart |
| Drawn chart, frozen model (cell 15) | 56 tokens, 8.8 s; header `Quarter \| Quarterly revenue`; cell accuracy 0.9, title match 1.0 | cell accuracy 0.9 |
| One extraction (51 tokens), CPU | 7.3 s at 24 threads; 27.2 s at 2 threads | — |
| One adaptation step (batch 4, 2 blocks) | 11.8 s at 24 threads; 18,879,744 / 282,285,696 trainable | same counts |
| Whole default path | not run (shard not downloaded; T4 record 4,983 s) | 11/11 post-restart cells |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (DPC-M1).** Both recorded passing runs stopped at the install cell's stale-module guard and passed only after a kernel restart; the release record calls that expected.
2. **Re-running from the documented cells reuses the already adapted model (DPC-M2).** The BYOD instruction ("re-run from that cell") and the optional experiments (edit cell 19 and re-run) start from the SynthChartNet-adapted weights, label them "frozen model", and the `TRAINABLE_DECODER_LAYERS = 1` experiment exports an adapter that cannot reproduce the model (Section 9 fails with a bare `AssertionError`).
3. **Guided layer mostly absent (DPC-M3).** Declared `GUIDED`; no audience statement, how-to-use, roadmap, glossary, prediction prompts, checkpoints, troubleshooting or conclusion scaffold, and 78,540 characters of carried package code are not marked as infrastructure.
4. Minor issues: no runtime planning information for a CPU default that would take hours (DPC-m1), a BYOD contract that understates its real limits (DPC-m2), a result that is never interpreted for this run (DPC-m3), and no chart ever shown to the learner (DPC-m4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`, References) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated as such. Knowledge prerequisites: basic Python and PIL; encoder–decoder generated tokens; chart data-table layout; that a well-formed table is not a correct one; what baselines that never see the image show |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; "runs on CPU (float32)", CUDA used when present, GPU recommended for Sections 6–8. `tutorials/README.md` lists **CPU** as the default runtime |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; digest-pinned SynthChartNet shard converted, validated and split image-disjoint by chart type; inference contract with manifest and refusal; three baselines and frozen scores per chart type; bounded fine-tuning with explicit hyperparameters and validation epoch selection; held-out evaluation; drawn chart re-extracted; safetensors adapter export and reload parity; BYOD zip through "the same validation, seeded image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `ddf0cc4` |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code; cells 5, 7 and 9 carry `pipeline.py`, `metrics.py`, `samples.py`). Also read: `adapt`, `evaluate`, `save_artifact`, `load_artifact`, `split_dataset`, `validate_dataset`, `_check_record`, the baselines; `tools/build_notebook.py`, `tools/notebook_template.py`; `tests/test_adaptation_model.py`; `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md`: Kaggle Tesla T4 v4, 2026-09-26, **the reviewed blob**, clean Hugging Face cache, default path. Pass 1 (189.3 s) stopped at the install cell's stale-module guard; the kernel was restarted; pass 2 ran 11/11 (4,793.7 s). Frozen test cell accuracy 0.160 / RNSS 0.448 / exact 0.006; baselines 0.000 / 0.007 / 0.037; adapted 0.181 / 0.447 / 0.000; best epoch 2; reload parity 8/8. The executor's raw output is not archived locally; this review relies on the record. No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 24 threads, build venv `dimer-next16` with exactly the notebook's `PINS` (Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, pyarrow 25.0.1). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Not a clean runtime:** the 8 snapshot files were hard-linked from the local clone into a scratch working directory; cell 11 wrote the manifest fresh and `verify_snapshot` re-hashed every file. The SynthChartNet shard was **not** downloaded, so Section 4's default branch, Sections 6–9 at default scale and the full default path were **not** executed here.
  - **Executed:** cells 3, 5, 7, 9, 11, 15 on the real model (P1, P2); CPU cost of extraction at 24 and 2 threads and of one adaptation step on PIL stand-in charts (P3, labelled estimate); re-run semantics of `adapt`/`save_artifact`/`load_artifact` on a tiny random Pix2Struct built like `tests/test_adaptation_model.py` (P4, P4b); cell 13's BYOD branch through a `google.colab` upload shim with 8 zips plus a no-Colab case, the split minimum and the evaluation ceiling (P5, P5k). Two invocations, 184.9 s and about 20 s wall.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong on the default path as recorded (T4 11/11 after restart; reload parity 8/8) and on the parts run here. Defects: the install pattern forces a restart (DPC-M1); a re-run of the BYOD or experiment cells silently continues from adapted weights and can export an adapter that does not reproduce the model (DPC-M2); BYOD rejects datasets the stated contract accepts (DPC-m2).
- **Promise fulfilment:** every default-path stage is implemented and was recorded. BYOD is promised to flow "through the same … baselines, fine-tuning, held-out evaluation" cells, but those cells then compare against an already adapted "frozen" model (DPC-M2) and need at least 50 records, not 8 (DPC-m2).
- **Scientific validity:** good. Image-disjoint split by declared id and byte digest; validation selects the epoch; test used once; three non-neural baselines; RNSS reported beside the position-wise metric with the layout caveat stated. The run's own outcome (cell accuracy +0.021, RNSS −0.001, exact match 0.006 → 0.000) is exactly the pattern the notebook warns about, but the notebook never states it for this run (DPC-m3). Re-runs break the "frozen" reference (DPC-M2).
- **Learner experience:** thorough stage prose and two "Look for" notes; strong limits section and "carry to real data" advice. No chart is ever displayed (DPC-m4); no runtime estimate or progress feedback for steps that take tens of minutes on T4 and hours on CPU (DPC-m1); guided layer mostly absent (DPC-M3).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6 (DPC-M1); DAT12, DAT19, VAL6 (DPC-m2); REL12 (no BYOD evidence recorded); UX12 (DPC-m1, the CPU claim "each extraction costs seconds" is unqualified). SHOULD deviations: GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14, UX8 (DPC-M3); GDL10, UX10 (DPC-M2); RUN12 (DPC-m1); EXE2 (DPC-m2); GDL8, UX3, UX11 (DPC-m4). Declared spec 2.0 (DPC-S1).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all`, "no configuration edit" | cell 3 in-kernel `pip install` + stale-module guard | Kaggle v4 pass 1 stopped at cell 3 (189.3 s), restart, pass 2 11/11 | Section 1: "the cell stops with a restart instruction" | **Not met** (DPC-M1) |
| Pinned, digest-verified snapshot; no Hub font | cell 11 | P1: 8/8 verified at `6e76d624…`, VQA processor, Pillow font | clear | Met |
| Pinned shard, OTSL → DePlot target, stratified image-disjoint split | cell 13 | record: 14,568 rows, 360/80/160, no shared image | well explained | Met (documented) |
| Four dataset refusal probes | cell 13 | record: each rejected | listed | Met (documented) |
| Inference contract on a drawn chart; no score | cell 15 | P2: 56 tokens, `Quarter \| Quarterly revenue` header, 0.9 / 1.0 (matches the stated smoke) | "No score exists"; verdict `sample-sanity` | Met |
| Three baselines and frozen model, per chart type | cell 17 | record: 0.000 / 0.007 / 0.037 vs 0.160 | metric caveats stated | Met (documented) |
| Bounded fine-tuning, trainable set and hyperparameters explicit | cell 19 | P3: 18,879,744 / 282,285,696; record best epoch 2 | clear; epoch-0 row labelled "frozen model" | Met on first run; mislabelled on re-run (DPC-M2) |
| Held-out evaluation with comparison; `adapted_beats_frozen` | cell 21 | record: +0.021 cell accuracy, −0.001 RNSS, exact 0.006 → 0.000, flag `True` | reading order given; this run's result not interpreted | Computation met, conclusion left to the learner (DPC-m3) |
| Drawn chart after adaptation; adapter export; fresh reload parity | cell 23 | record: drawn 0.9, 29 tensors, parity 8/8 | explained | Met on the default path; fails after the `TRAINABLE_DECODER_LAYERS = 1` re-run (DPC-M2) |
| BYOD through the same stages | cell 13 BYOD branch, then cells 15–23 | P5: <50 records rejected with an unnamed split; >2,500 rejected in Section 6; P4b: re-run continues from adapted weights | contract stated as 8..5,000 | **Partly met** (DPC-M2, DPC-m2) |
| Optional experiments | Interpretation section | P4b: re-running cell 19 trains on top of the adapted model; 1-block export loses the other block | outcomes presupposed | **Not met as written** (DPC-M2, DPC-m3) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install the pinned runtime; read the carried modules | run cells | versions printed; 78,540 characters of code with no reading guidance (DPC-M3) |
| Stage and verify the snapshot | run cell | verified count and identity printed |
| Download, convert, validate and split the shard | run cell, read output | one OTSL table beside its target printed; strong |
| Read `text`, `table`, `new_tokens`, `truncated` | run cell 15 | fields printed; the chart itself never shown (DPC-m4) |
| Score frozen model vs baselines per chart type | run, read dicts | no prediction prompt, no checkpoint (DPC-M3) |
| Bounded fine-tuning with validation selection | run cell 19 | epoch history printed |
| Evaluate on the held-out split | run cell 21 | no prompt to state the run's result (DPC-m3) |
| Export and reload an adapter | run cell 23 | parity asserted |

Objectives are phrased as actions the code performs (GDL5 partly met); none is followed by a check of the learner's understanding.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 25 cells | Each section says what runs, why, and how to read it; the no-score, layout and dispersion caveats are clear and repeated. Friction: the notebook never shows a chart, so the learner reads tables of images they cannot see (DPC-m4); the CPU default gives no duration (DPC-m1); Sections 2's three cells are 78,540 characters of package code with no "you may skip this" (DPC-M3); the interpretation is generic and never states this run's outcome (DPC-m3). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (partial, CPU) | Kaggle v4: pass 1 stopped at the install guard, pass 2 11/11 after a restart, 4,983 s total (DPC-M1). Direct: setup cells and Section 5 on the real model reproduce the recorded drawn-chart result; Sections 4 and 6–9 at default scale not executed. CPU estimate from P3: at least 2.4 h at 24 threads and at least 5.6 h of extraction alone at 2 threads (lower bounds; stand-in tables are shorter than SynthChartNet's). No Colab run. |
| **Active learning** | Direct, tiny random model (P4b); real-model effect inferred | Following the Interpretation section literally — edit cell 19 and re-run it: `adapt` starts from the current (already adapted) weights, records epoch 0 as `"frozen model"`, and with `TRAINABLE_DECODER_LAYERS = 1` exports only the last block; reloaded onto a fresh base, the block trained by the first run reverts to base (max abs diff 0.006 vs in memory), and reload parity is 0/4 tables, which in cell 23 is a bare `AssertionError`. Not run on the real model (would need the default run first). |
| **Reuse and recovery** | Direct (P5, P5k), Colab upload dialog not verified (shim) | BYOD zip via upload shim. Refused with the rule named: target over 512 characters (record named), image in a zip subfolder referenced by path (`image file not found: work\byod\images\0.png`, cause not explained). Refused unclearly: 12–49 records → `ValueError: 2 records; 8..5000 are required` (the test split, not named); no `records.jsonl` → bare `StopIteration`; not a zip → `BadZipFile`; non-Colab runtime → `ModuleNotFoundError: google.colab`. Accepted: 50+ records (test 10 / val 8 / train 32). Traversal member names are flattened and written inside `work/byod` (safe). At 2,600 records the split passes but Section 6's `pipe.evaluate` refuses `520 records; 1..500 are required` after Section 5 has run the model. After a default run, BYOD re-run compares against the SynthChartNet-adapted model as "frozen" (P4b `evaluate()['adapted'] = True`) (DPC-M2). |

## 5. Findings

### Major

#### DPC-M1 — `Run all` needs a manual restart after the install cell

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, `_INSTALL_GUARD` lines 47–70: `pip install` at line 61, `RuntimeError` at line 69; Section 1 prose at line 453); `docs/release-verification.md` step 4 and both recorded T4 runs; `tutorials/README.md` Run-all column.
- **Observed issue:** the cell `pip install`s nine pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. The release procedure says "an interpreter restart after the install is expected", and the registry reports "passed — 11/11 post-restart code cells".
- **Consequence:** a learner selecting **Run all** on a stock hosted image hits an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this, and a restart-dependent run is not a `Run all` PASS.
- **Evidence:** documented — Kaggle T4 v4 on blob `90d9093c`: "pass 1 189.3 s stopped at the install cell with the expected stale-module guard; kernel restarted; pass 2 4793.7 s" (v3 the same). Source — probe static `pip_install_in_kernel: true`, `uses_uv: false`.
- **Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab GPU runtime completes every code cell in a single **Run all** with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/deplot_chart_colab.ipynb` returns nothing; step 4 of the procedure no longer calls a restart expected.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### DPC-M2 — Re-running the BYOD or experiment cells reuses the adapted model, mislabels it "frozen", and can break the export

- **Cell/section:** opening cell ("set `USE_BYOD = True` in Section 4 and re-run from that cell"), cell 13 (no pipeline reload), cell 17 ("frozen model"), cell 19 (`pipe.adapt`), cell 23 (`assert parity[...]` with no message), "Optional experiments" in the Interpretation section. Generator: `tools/notebook_template.py` line 39 (BYOD instruction), lines 340–356 (adaptation cell, from `EPOCHS` at line 341), line 450 (bare assert), lines 503–506 (experiments). Module: `pipeline.py` `adapt` (`initial_state` taken from the current weights, epoch 0 noted `"frozen model"`), `save_artifact` (writes only the current run's trainable tensors).
- **Observed issue:** `pipe` is built once in cell 11 and mutated in place by `adapt`. (a) BYOD after a default run: cells 15–23 re-run on the SynthChartNet-adapted model, so Section 6's "frozen model" scores, `frozen_beats_*`, the epoch-0 "frozen model" row and `adapted_beats_frozen` all compare against an adapted model; `pipe.evaluate` itself records `adapted: true` in that "frozen" result. (b) Experiments: editing `LEARNING_RATE` or `TRAINABLE_DECODER_LAYERS` and re-running cell 19 continues training the adapted model; with `TRAINABLE_DECODER_LAYERS = 1` the exported adapter holds only the last block, the reload overlays it onto a fresh base, the block changed by the first run reverts, and cell 23's bare `assert` fails. No cell says which cells to re-run or that the runtime must be reset.
- **Consequence:** the learner's BYOD comparison and the notebook's suggested experiments produce numbers labelled as frozen-versus-adapted that are not, and the layer-count experiment ends in an unexplained crash. These are the notebook's only active-learning paths.
- **Evidence:** direct (P4b, tiny random Pix2Struct, lr 1e-3, final-epoch selection): after run 1 `evaluate()['adapted'] = true`; run 2's epoch 0 recorded as `"frozen model"`; run 2 kept run 1's block `decoder.layer.1` (`run2_started_from_run1_state: true`); artifact tensors only block `2`; reloaded `decoder.layer.1` equals base, max abs diff to the in-memory model 0.006; reload parity 0/4. (P4, with validation selection, kept epoch 0 on the random model, so it shows the labels but not the state change.) On the real model the parity failure is inferred: the default run moves held-out cell accuracy by 0.021, so its block-10 changes are not negligible.
- **Recommended correction:** make cells 17 and 19 self-contained: construct a fresh `DePlotPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` for the frozen evaluation and for each adaptation (or refuse to `adapt`/score "frozen" when `pipe.adapter is not None`, with a message naming the cells to re-run); state in the BYOD paragraph and each experiment exactly which cells to re-run; give the reload assertion a message naming the mismatch and the next step.
- **Acceptance check:** after a default run, setting `TRAINABLE_DECODER_LAYERS = 1` and re-running only cell 19 either trains from the base checkpoint (cell 23 then passes) or stops with a message naming the cells to re-run; after a default run, the BYOD re-run's Section 6 result has `adapted: false`; every experiment line names its field and the cells to re-run.
- **Spec:** GDL10, UX10, UX5, DAT13, DAT14, FT5.

#### DPC-M3 — Declared `GUIDED`, but most of the guided layer is absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, 9, 11, end of notebook. Generator: `tools/notebook_template.py`, `tools/build_notebook.py`.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap (the **Run all** paragraph is one 160-word sentence), no Input → Model → Output contract, no glossary (OTSL, linearised table, `<0x0A>`, Pix2Struct patches, VQA header rendering, teacher forcing, RNSS, medoid, cell accuracy, adapter), no prediction prompts before Sections 6, 7 or 8, no interpretation checkpoints, no troubleshooting (restart, 1.6 GB of downloads, CPU duration, out of memory, BYOD errors), no conclusion scaffold. The three module cells (78,540 characters) and the manifest and install cells are not titled **Infrastructure** and are not collapsed (`cellView` absent everywhere). The notebook does have two "Look for" notes, metric-reading guidance in Sections 6 and 8 and a strong limits section.
- **Consequence:** a self-paced learner gets an accurate, well-explained reference workflow but no prompt to commit to a prediction, check understanding or recover from expected failures, and must scroll past the package source without being told it can be skipped.
- **Evidence:** source; probe static `guided_markers` all `false` (How to use / Roadmap / Glossary / Check your reasoning / What to notice / Expected result / Troubleshooting / conclusion / intended learner / Infrastructure / Input →), `cellView_form_cells: []`, `look_for_count: 2`, `carried_module_chars` 41,389 + 11,411 + 25,740.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC §25.13's reference notebook: audience and how-to-use, roadmap, task contract, glossary, a prediction before Sections 6, 7 and 8 (for example "will the frozen model beat the medoid table? will RNSS rise with cell accuracy?"), a collapsible checkpoint after the comparison, a Predict → Change one thing → Run → Observe → Explain activity built on DPC-M2's fix, troubleshooting, and a conclusion scaffold; title cells 3, 5, 7, 9 and 11 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`; cells 3, 5, 7, 9 and 11 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14, UX8.

### Minor

#### DPC-m1 — No duration or progress for a default path that takes 80 minutes on T4 and hours on CPU

- **Cell/section:** opening cell ("the CPU path works but is slow … timings … are recorded in `docs/release-verification.md`"), Prerequisites ("each extraction costs seconds on CPU"), cells 17, 19, 21; `tutorials/README.md` "Default runtime: CPU". Generator: `tools/notebook_template.py` lines 35 and 127.
- **Observed issue:** the notebook states no duration. The only measured default run took 4,983 s on a T4 (adaptation 2,290 s). The registry names CPU as the default runtime, yet the default path performs 738 extractions and 270 training steps. `pipe.evaluate` prints nothing until all 160 (or 80) charts are done, and `adapt` prints once per epoch.
- **Consequence:** a learner on the documented default (a hosted CPU runtime) cannot plan time, and sees no output for very long stretches; a free hosted session may end before the run does.
- **Evidence:** documented — T4 record. Direct (P3, estimate): 7.3 s per 51-token extraction at 24 threads, 27.2 s at 2 threads; 11.8 s per training step at 24 threads; lower bounds 2.4 h (24 threads) and 5.6 h of extraction alone (2 threads).
- **Recommended correction:** state the measured T4 time (with date and runtime) and a labelled CPU estimate in the Prerequisites; make a GPU runtime the documented default in the notebook and the registry (or shrink the CPU default); add per-N-chart progress lines in `evaluate`/`adapt`.
- **Acceptance check:** the Prerequisites give a measured GPU time and a labelled CPU estimate; `tutorials/README.md` names the runtime the evidence used; Sections 6–8 print progress at least every 20 charts.
- **Spec:** UX12, RUN12, ENV4, GDL13.

#### DPC-m2 — BYOD contract understates its real limits; several failures are unclear

- **Cell/section:** Prerequisites ("a dataset needs 8..5,000 records"), cell 13 BYOD branch (generator `tools/notebook_template.py` lines 160–180, prerequisites line 129).
- **Observed issue:** cell 13 re-validates every split with the default 8-record minimum, so a BYOD set needs **at least 50 records** (test 10 / val 8 / train 32 at 50); 12–49 records fail with "2 records; 8..5000 are required", which does not name the split. Above about 2,500 records the test split exceeds `MAX_EVAL_RECORDS` (500) and Section 6 fails ("520 records; 1..500 are required") after Section 5 has run the model. A zip without `records.jsonl`/`.json` raises a bare `StopIteration`; images in a zip subfolder referenced by path fail as "image file not found" because member names are flattened, without saying so; upload works only through `google.colab` (`ModuleNotFoundError` on Kaggle or Jupyter) and there is no path field; no expanded-size limit. No BYOD run is in the release record.
- **Consequence:** a user who follows the stated contract with 8–49 charts is rejected with a message that points at the wrong number; others hit errors that do not name the fix.
- **Evidence:** direct (P5, P5k, upload shim, PIL stand-in images): outcomes as listed; refused with a clear rule: target over 512 characters; traversal names flattened inside `work/byod` (safe).
- **Recommended correction:** validate the whole BYOD set once and the splits with `min_records=1` (or state and check the real minimum before splitting); check the test/validation sizes against `MAX_EVAL_RECORDS` in cell 13; raise a named error when no records file is found and when a record path contains a folder; add a `BYOD_ZIP_PATH` form field that bypasses `google.colab`; add an expanded-size cap; record one accepted and one refused BYOD run in `docs/release-verification.md`.
- **Acceptance check:** a 12-record BYOD zip either passes cell 13 or is refused before splitting with a message stating the real minimum; a 2,600-record zip is refused in cell 13; a zip without records and a non-Colab run without a path each stop with a message naming the fix; the release record holds the REL12 BYOD entries.
- **Spec:** DAT12, DAT19, VAL6, UX10, EXE2, REL12.

#### DPC-m3 — The run's result is never interpreted, and the experiments presuppose their outcome

- **Cell/section:** cell 21 output, Interpretation section, "Optional experiments". Generator: `tools/notebook_template.py` lines 386 and 477–506.
- **Observed issue:** the recorded run gives cell accuracy +0.021, RNSS −0.001 and exact-table match 0.006 → 0.000, with `adapted_beats_frozen: True`. That is the "layout learning, not better reading" pattern the notebook describes, but the notebook only prints the flag; the Interpretation section is fixed text that never asks the learner to state this run's outcome. The experiments tell the learner what will happen ("watch the training loss fall while the validation cell accuracy drops and the selector keeps an early epoch") rather than asking.
- **Consequence:** the headline `True` invites "adaptation works"; the learner is not led to the conclusion the evidence supports.
- **Evidence:** documented T4 metrics; source.
- **Recommended correction:** after cell 21, print a one-line reading of the three deltas and add a prompt (with a collapsible sample answer) asking whether the gain is reading or layout; phrase the experiments as questions.
- **Acceptance check:** cell 21 prints all three deltas with a sentence keyed to their signs; the Interpretation section contains a prompt to state the run's cell-accuracy and RNSS change; no experiment line states its outcome.
- **Spec:** EVAL15, GDL7, GDL8, GDL14.

#### DPC-m4 — The notebook never shows a chart

- **Cell/section:** cells 13, 15, 21, 23.
- **Observed issue:** no code cell displays an image (no `display`, `plt` or `.show`). The drawn bar chart is described in prose; the SynthChartNet examples are printed only as truncated table strings.
- **Consequence:** in a chart-reading tutorial the learner cannot compare a prediction with the chart it came from, which is what "read the per-chart-type breakdown" and "a transposed target scores as a miss on every cell" require.
- **Evidence:** source; probe static (no display calls outside the carried modules).
- **Recommended correction:** display the drawn chart in cell 15 and, in cells 17 and 21, one chart per type beside its target, frozen and adapted tables.
- **Acceptance check:** the executed notebook shows the drawn chart and at least one SynthChartNet chart per chart type next to its tables.
- **Spec:** UX3, UX11, GDL8.

### Suggestions

- **DPC-S1** — Declare `notebook_spec` 2.2 (currently 2.0 in metadata, the opening cell, `NOTEBOOK_SOURCE`, References and `tutorials/README.md`) once the guided layer lands.
- **DPC-S2** — Add an RNSS-based flag beside `adapted_beats_frozen` so the export records both readings.
- **DPC-S3** — Record per-stage wall times in `deplot_chart_result.json` (adaptation dominates: 2,290 of 4,794 s on T4).
- **DPC-S4** — Say in Section 4 that the 516 MB shard is the large download and offer a smaller pinned subset for CPU users.

## 6. Readiness

**Needs revision.** Open Majors DPC-M1 to DPC-M3. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (RUN1/RUN10), a recorded re-run of one experiment and of BYOD after a default run, and the REL12 BYOD exercise (one compatible and one incompatible input) in `docs/release-verification.md`.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, model staged, labelled above):** setup cells and the Section 5 inference contract on the real model (matches the record); CPU costs per extraction and per step; the re-run semantics on a tiny random model (state carried over, "frozen model" label, 1-block export losing the other block, parity 0/4); the BYOD outcomes and limits in §4.
- **Verified from documented evidence:** the restart on both T4 runs; the T4 metrics, timings and reload parity.
- **Inferred from source:** that the real-model `TRAINABLE_DECODER_LAYERS = 1` re-run fails cell 23 (it follows from P4b and the size of the default run's change); that the Colab upload dialog delivers files as the shim did; the CPU duration beyond the measured lower bounds.
- **Not verified:** any Colab run; Section 4 and Sections 6–9 at default scale in this review; BYOD with real charts.
- **Most likely to be wrong:** DPC-M2's real-model parity failure — the reload mismatch is shown on a tiny random model; on the real checkpoint, greedy tables could coincide on the eight parity charts even with block 10 reverted, in which case the visible harm is the wrong "frozen" labels and a silently non-reproducing adapter rather than a crash. The finding stands as Major either way because the comparisons the learner reads are mislabelled.
