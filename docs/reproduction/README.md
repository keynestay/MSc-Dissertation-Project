# Reproducing The Experiments

Use the repository root as the working directory for commands. [Environment notes](../../environments/README.md) describe the JAX and GPflow dependencies. Every notebook begins with a setup cell that resolves the repository root, adds it to Python's import path, and changes into its own output directory under `generated/`. Run that cell first. For a new run, restart the notebook kernel and run all cells in order.

## Bernoulli Experiment

Use [the Bernoulli instructions](../../experiments/bernoulli_abc/README.md), then run `visualize_abc_results.ipynb`. The plotter selects the dense-checkpoint run confirmed for Figure 2.

## Soay Data And Posterior Samples

For each stage, run the baseline notebook before the GP notebook:

| Stage | Baseline | GP Dataset | Posterior Sampling |
| --- | --- | --- | --- |
| Size only | `size_only/data_generation/Baseline_IBM_SOAYSHEEP.ipynb` | `size_only/data_generation/GP_IBM_final.ipynb` | `size_only/inference/GP_df_inference.ipynb` |
| Age size | `age_size/data_generation/Baseline_IBM_agesize.ipynb` | `age_size/data_generation/GP_IBM_agesize.ipynb` | `age_size/inference/GP_df_inference_agesize.ipynb` |

These paths are under `experiments/soay/`. Data and true IPM metrics go into `generated/soay/<stage>/data/`; posterior samples go into `generated/soay/<stage>/inference/`. The size-only GPU inference alternative is `python -m experiments.soay.size_only.inference.run_gp_df_inference --output-dir generated/soay/size_only/inference`. The corresponding SLURM launcher retains its cholgpu01 allocation.

## Train The Surrogates

Age-size training accepts the recovered final settings:

```bash
python -m experiments.soay.age_size.training.train --min-locations 1 --max-locations 512 --num-steps 2000000 --checkpoint-interval 100000 --validation-interval 400000 --validation-steps 10000 --batch-size 8 --seed 23 --wandb-mode disabled
```

The final size-only recorded settings are in `configs/size_transformer_recorded.json`. The current trainer is `python -m experiments.soay.size_only.training.train_transformer_deeprv --help`. Use min/max locations 1/1024, two million steps, checkpoint interval 100,000, validation interval 400,000, 10,000 validation steps, batch 8, seed 23, and size Normal mean 3 / SD 0.5. **Resolve the historical lengthscale parameterisation in [known gaps](gaps.md) before claiming a reproduction of job 66235.** The current trainer's defaults are not a verified copy of that run.

Transformer checkpoints go into `generated/soay/<stage>/training/transformer/`. The prior-calibration notebooks in each `training/` folder document the relevant priors.

For fixed-location gMLP, run `experiments/gmlp_mcmc/Train_gMLP_DeepRV_Soay_760.ipynb`, then `deeprv_MCMC_inference.ipynb`. The training function creates `generated/gmlp_mcmc/training/gmlp_deeprv_rbf_soay_760/`; the inference notebook now reads that same directory. Its metric-analysis source for Table 6 remains missing.

## Full ABC And IPM Metrics

Run the GP ABC first to regenerate summary selection and baseline particles, then DeepRV ABC with its checkpoint and selection file:

```bash
python -m experiments.soay.size_only.abc.tracked.run_soay_abc_hpc generated/soay/size_only/data/gp_ibm_dataset.csv generated/soay/size_only/inference/gpdf_gp_samples.npz generated/soay/size_only/abc/gp
python -m experiments.soay.size_only.abc.tracked.run_soay_abc_hpc_deeprv generated/soay/size_only/data/gp_ibm_dataset.csv generated/soay/size_only/inference/gpdf_gp_samples.npz generated/soay/size_only/training/transformer/model.ckpt generated/soay/size_only/abc/deeprv generated/soay/size_only/abc/gp/run_info.json
```

Age-size equivalents are `experiments.soay.age_size.abc.gp.run_soay_abc_hpc_agesize` and `experiments.soay.age_size.abc.deeprv.run_soay_abc_hpc_deeprv_agesize`. They take the corresponding age-size dataset, posterior archive, output directory, and, for DeepRV, checkpoint and selection JSON. `configs/<stage>_abc_selection.json` preserves the historical chosen statistics; it can be supplied as the selection JSON where the runner accepts that argument.

After both methods complete, run `experiments/soay/<stage>/abc/Make_IPM*.ipynb` for population/IPM metrics. These notebooks also read GP/GLM posterior samples and true metrics from the earlier generation stages.

## IBM Validation And Reduced ABC

Both stages have `validation/ibm/make_design.py`, `run_exact_gp_cpu.py`, and `run_deeprv_gpu.py`. Launch these as modules from the root. Each CLI provides `--help`; pass data, samples, shared design, and output paths under `generated/soay/<stage>/validation/ibm/`. Use the settings in `configs/<stage>_validation.json`: 5,000 parameter sets, 50 replicates, and 100 results per chunk. Size-only comparison then uses `validation/ibm/compare_results.py`.

After simulation and comparison, run the analysis notebook under `validation/reduced_abc/`, followed by `Offline_ABC_IPM_Point_Estimates.ipynb`. Both read the same canonical Transformer validation folder. The size-only notebook no longer switches to the older benchmark halfway through.

## Covariance, KS, And Speed

The covariance and KS folders contain evaluation modules, SLURM launchers, and plot notebooks. Launchers accept `REPO_DIR` and `OUTPUT_DIR` overrides, retain GPU resources and `srun --cpu-bind=none`, and protect existing output folders. Size covariance generates nested draw counts and the final plot uses 500 draws; age-size covariance explicitly requests year 2. Run the corresponding plotting notebook after each evaluation.

The benchmark folders contain `make_design`, CPU and GPU simulation modules, and timing comparison modules. `configs/size_speed.json` and `configs/size_scaling.json` preserve recovered designs. Scaling uses each initial population 250/500/750/1,000, 100,000 datasets, one year, observed years 1–4 for initial sizes, and DeepRV batch 8. Run `compare_scaling` on those output directories, then `plot_scaling_curve.ipynb`. The original Table 13 result record remains missing.

Full regeneration is computationally expensive and has not been executed as part of this organisation. Outputs stay ignored by Git. See [the verification record](../cleanup/local_organisation_2026-10-10.md) for checks actually performed.
