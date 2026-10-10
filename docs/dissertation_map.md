# Dissertation Experiment Map

| Dissertation Item | Retained Source | Selected Settings |
| --- | --- | --- |
| GP illustrations | `experiments/background/Draw_RBF_samples.ipynb` | Notebook parameters |
| Figure 2, Bernoulli ABC | `experiments/bernoulli_abc/` | Dense checkpoints, job 68372, 256 locations, two blocks |
| gMLP training and posterior sampling; Table 6 partly recovered | `experiments/gmlp_mcmc/` | 760 fixed locations; metric evaluation remains missing |
| Size-only Soay model and ABC | `experiments/soay/size_only/` | Main GP generator, Transformer job 66235 |
| Figure 5, size-only covariance | `size_only/validation/covariance/` | Run 73309, year 1, 500-draw plot from nested 50/100/250/500 draws |
| Age-size Soay model and ABC | `experiments/soay/age_size/` | Main GP generator, Transformer job 66312; all Try Predict_y variants omitted |
| Figure 8, age-size covariance | `age_size/validation/covariance/` | Regenerate year 2; old record pooled years 2–4 |
| Figures 9, 11, 12, size-only validation | `size_only/validation/ibm/` and `size_only/validation/reduced_abc/` | Final Transformer comparison, 5,000 parameter sets × 50 replicates |
| Age-size validation | `age_size/validation/ibm/` and `age_size/validation/reduced_abc/` | 5,000 parameter sets × 50 replicates, 10 years |
| Table 12, size-only speed | `size_only/benchmarks/` | 50,000 datasets, four years, DeepRV batch 8, GPflow 48 CPUs |
| Scaling experiment | `size_only/benchmarks/` | 100,000 datasets, one year, initial populations 250/500/750/1,000 |
| Table 13, age-size speed | `age_size/benchmarks/` | Original timing JSON unavailable |

Relative paths beginning with `size_only/` or `age_size/` are under `experiments/soay/`. Run names containing `1_1024` or `1_512` describe the range of input locations. Both recorded final Transformer models have four blocks, dimension 128, and 512 random Fourier features.
