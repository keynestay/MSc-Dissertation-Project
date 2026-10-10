# Experiment Settings

These small records preserve model and run settings before generated files are removed. Paths point to `generated/`; W&B tracking is disabled in portable copies. They are reference records, not automatically loaded by every script. Commands and notebook order are in [the reproduction guide](../docs/reproduction/README.md).

`size_transformer_recorded.json` contains historical `lengthscale_mean` and `lengthscale_variance` fields. The current trainer accepts `lengthscale_log_mean` and `lengthscale_log_sd`; their relationship has not been verified. Do not silently substitute values. The original record remains unchanged under `docs/reproduction/training_runs/`.

`age_size_covariance.json` separates the author-confirmed year 2 from the historical pooled-years record. ABC selection files preserve the selected statistic names needed by the DeepRV runners, without storing ABC results.

## Sources

- `bernoulli_dense.json`: `dl4bi/DEEPRV_ABC_TEST/results/job_68372_dense_checkpoints/config.json`
- `size_transformer_recorded.json`: `docs/reproduction/training_runs/run-20260817_214840-1dci4r39/training_parameters.json`
- `age_size_transformer.json`: `dl4bi/SOAYSHEEP/transformer_deeprv_age_size_1_512_2m_job_66312/metadata.json`
- `gmlp_760.json`: `SOAYSHEEP/Train_gMLP_DeepRV_Soay_760.ipynb and metadata_soay_760.json`
- `size_only_abc_selection.json`: `dl4bi/SOAYSHEEP/ABC_results/PRIOR_tracked_gp_opt6_year1_50k/run_info.json`
- `age_size_abc_selection.json`: `dl4bi/SOAYSHEEP/ABC_results/ABC_agesize/PRIOR_full_opt6_50k/run_info.json`
- `size_only_validation.json`: `dl4bi/SOAYSHEEP/DeepRV_Evaluation/benchmark_5k_r50_transformer_1_1024/`
- `age_size_validation.json`: `dl4bi/SOAYSHEEP/DeepRV_Evaluation/benchmark_age_size_5k_r50_transformer_1_512/`
- `size_covariance.json`: `soay_dowson_landau_draw_counts_73309/run_metadata.json`
- `age_size_covariance.json`: `dl4bi/SOAYSHEEP/transformer_dowson_landau_evaluation_age_size/soay_age_size_dowson_landau_job_73379/run_metadata.json`
- `size_speed.json`: `DeepRV_SPEED`
- `size_scaling.json`: `DeepRV_Evaluation/empirical_scaling_reverse_100k_ada/n_250`
