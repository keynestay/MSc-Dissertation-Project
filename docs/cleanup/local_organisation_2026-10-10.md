# Local Organisation — 10 October 2026

The author authorised a lean, source-only cleanup of dl4bi and the nine previously agreed dissertation folders on the Desktop. Generated artifacts and unused variants were removed after source and required settings were protected. No files were uploaded, staged, or committed. Git history, HEAD, index, configuration, and packed refs were preserved.

## Retained Layout

`dl4bi/` holds the shared library. `experiments/` groups the Bernoulli experiment, gMLP/MCMC, and Soay size-only/age-size workflows. `configs/` holds small reproduction settings; `environments/` holds dependency records; `docs/` explains the run order, dissertation mapping, and unresolved provenance. All 26 retained notebooks have empty outputs and execution counts. Regenerated artifacts use an ignored `generated/` folder.

128 original source/configuration files were retained; four identical age-size provenance copies were consolidated, and older source variants were omitted. Source moves and original hashes are recorded in `source_moves_2026-10-10.json`. Complete old selected/review source backups and removal journals were retained temporarily outside iCloud at `/private/tmp/dl4bi_final_cleanup_7_jgqd4i/`; this is temporary local storage, not part of the repository.

## Changes And Checks

- Existing numerical/model code was preserved. A comparison of 117 retained Python/notebook sources found no changes outside imports and file-location expressions. The two changed path-search cells were also reviewed directly.
- The three previously authorised notebook repairs remain: the incomplete import was removed, `plt.show()Make` was repaired, and age-size baseline CSV exports were enabled.
- References now use the dense Bernoulli checkpoints and final Transformer validation. The age-size covariance launcher explicitly requests year 2, while the old years 2–4 record is documented separately.
- gMLP inference now reads the actual nested directory created by its training function. Historical size-only prior fields were preserved without guessing a conversion.
- The `.gitignore` permits source launchers and environment records while excluding regenerated artifacts, checkpoints, logs, and caches. Package configuration includes `experiments`, dissertation dependencies, and both locally checked and historically recorded JAX versions.
- 10 existing Bernoulli tests passed using Python 3.11/JAX 0.8.3. Small relocated training/model/covariance checks passed. GPflow 2.9.0, ABC_GP_IPM, and both GP ABC modules imported successfully in the existing abcgp environment.
- All 116 Python files and 273 code cells across 26 notebooks parsed. Notebook-format validation passed. All six shell scripts passed `bash -n`, with GPU requests and `srun --cpu-bind=none` preserved. All relocated local imports resolve. The module CLI and `git diff --check` passed.
- Every listed removal and preserved Git metadata hash was verified. Existing Git-eligible files contain no generated arrays, datasets, checkpoints, figures, dissertation PDFs, or logs.

Full training, MCMC, ABC, GPU validation, and benchmarks were not rerun. Fresh dependency installation and a combined JAX/GPflow ABC environment remain unverified. See [remaining reproduction gaps](../reproduction/gaps.md).

## Removal Totals

This round removed 31,355 file/link entries representing 6,195,655,787 logical bytes. The working tree is approximately 1.60 MB before these small final audit records. Git history remains approximately 1.28 GB. Logical totals do not measure physical disk space reclaimed from iCloud placeholders.

| Original Root | Entries Removed | Logical Bytes |
| --- | --- | --- |
| `ABC_results` | 47 | 16181965 |
| `DeepRV_Evaluation` | 10079 | 727406494 |
| `DeepRV_SPEED` | 7 | 7223322 |
| `dl4bi` | 21077 | 3842455765 |
| `soay_dowson_landau_73275` | 76 | 1439328963 |
| `soay_dowson_landau_draw_counts_73309` | 25 | 71059054 |
| `soay_dowson_landau_exact_gp_73281` | 7 | 31621967 |
| `transformer_deeprv_age_size_1_512_2m_job_66312` | 37 | 60378257 |

All nine agreed Desktop sibling folders are now removed. The first-round audit remains in this directory; its totals are separate from this round.
