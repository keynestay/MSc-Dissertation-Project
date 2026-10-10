# Dissertation Code: DeepRV For Bayesian Inference

Source code and generators used for the dissertation, organised by experiment. Generated datasets, checkpoints, posterior samples, metrics, figures, notebook outputs, W&B logs, and the dissertation PDF are excluded.

| Folder | Contents |
| --- | --- |
| `dl4bi/` | Shared neural-network and DeepRV library |
| `experiments/background/` | GP illustration notebook |
| `experiments/bernoulli_abc/` | Bernoulli GP training, ABC, and plotting |
| `experiments/gmlp_mcmc/` | Fixed-location gMLP training and posterior inference |
| `experiments/soay/common/` | Shared Soay IBM/IPM functions |
| `experiments/soay/size_only/` | Size-only generation, training, inference, ABC, validation, and benchmarks |
| `experiments/soay/age_size/` | Age-size versions of the same stages |
| `configs/` | Preserved settings for the selected experiments |
| `environments/` | Installation notes and recorded dependencies |
| `docs/` | Reproduction order, dissertation mapping, remaining gaps, and cleanup records |
| `tests/` | Existing library and Bernoulli checks |

Run commands from the repository root using `python -m experiments...`. Open notebooks with an environment containing their dependencies. Each notebook starts with a path setup cell and writes to an ignored `generated/` directory.

Start with [installation notes](environments/README.md) and [the reproduction guide](docs/reproduction/README.md). The [dissertation map](docs/dissertation_map.md) identifies the final runs; [known gaps](docs/reproduction/gaps.md) state which historical results cannot yet be reproduced exactly.

The shared DL4BI library derives from [MLGlobalHealth/dl4bi](https://github.com/MLGlobalHealth/dl4bi). Original Git history is retained. This local organisation does not stage, commit, or upload files.
