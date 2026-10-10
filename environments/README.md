# Python Environments

Use Python 3.11. DeepRV/JAX and GPflow have different NumPy requirements, so keep their environments separate for training and data generation. Mixed ABC workflows import both stacks and need an additional compatible combined environment; that installation has not been validated here.

For DeepRV on a CPU, install the local package with `python -m pip install -e ".[cpu,dissertation]"`. On a CUDA 12 cluster use `.[cuda12,dissertation]` instead. `pyproject.toml` includes the relocated `experiments` package and permits both the locally checked JAX 0.8.3 and historically recorded 0.10.2. This range is not a guarantee that every intermediate release has been tested.

`recorded_deeprv_requirements.txt` is the original cluster package list. Treat it as a historical record, rather than a portable macOS lock: it includes CUDA packages and platform-specific entries.

For the GPflow branch, `gpflow_requirements.txt` records the installed local GPflow/ABC versions. Install the notebook tools and scientific utilities as required: `python -m pip install -r environments/gpflow_requirements.txt pandas scipy scikit-learn matplotlib seaborn ray jupyterlab`. This recipe still needs a clean-environment installation check.

The external package is published as [`abc-gp-ipm==0.0.1` on PyPI](https://pypi.org/project/abc-gp-ipm/); its Python import name is `ABC_GP_IPM`. The published release supplies source and wheel distributions. Its installed local metadata pins GPflow 2.9.0. A separately documented upstream source repository was not found.

SLURM launchers retain the Oxford cluster environment activation and resource requests. Set `REPO_DIR` to your checkout and edit the cluster environment activation for another machine. Every retained JAX job uses `srun --cpu-bind=none`. Logs are excluded from Git.
