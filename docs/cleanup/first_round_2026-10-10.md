# First-Round Cleanup — 10 October 2026

Completed the approved first-round cleanup. Removed 785 regular files and 34 symbolic links, totaling 4,661,835,308 logical bytes (approximately 4.66 GB). This is the logical size of the deleted entries, including cloud placeholders, rather than a measurement of recovered disk space.

| Category | Entries Removed |
| --- | ---: |
| Logs and disposable caches | 229 |
| W&B output entries | 32 |
| Confirmed unused experiments, including `Try Predict_y` | 542 |
| Original W&B settings after verified preservation | 16 |

The preview contained 803 entries. Sixteen original settings files were additionally removed from W&B folders after their contents were copied and verified. Twenty-four small configuration, environment and launch records from eight original locations are retained under [recorded training settings](../reproduction/training_runs/README.md), with duplicates consolidated into five run folders.

Removed only the previewed logs, caches, W&B folders and unused experiments. Remaining dissertation datasets, checkpoints, generated results and unresolved source variants remain in place for the later source-only reorganization. The three previously authorized notebook repairs remain intact. No source code or notebook content was edited in this round.

Updated `.gitignore` to exclude runtime logs and caches, and to include the captured training requirements files. Existing Git history, index, configuration and all other pre-existing files were verified unchanged. No commit, upload or history rewrite was performed.

Verification checked all removal paths, byte checksums of 149 protected source/configuration files (allowing the recorded `.gitignore` update), checksums of all preserved settings, and filesystem metadata of all other inventoried files. The detailed [deletion and preservation manifest](first_round_2026-10-10.json) records every affected path.

Later work still includes the source-only folder reorganization, reference updates and reproduction instructions. Original gMLP population-metric code and the Table 13 timing JSON remain unlocated. The year-2 age-size covariance comparison still needs regeneration because the surviving historical run pooled years 2–4.
