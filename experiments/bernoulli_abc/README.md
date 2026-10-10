# Bernoulli GP And ABC

The submitted experiment uses the dense-checkpoint run: 256 locations, two gMLP blocks, seed 42, and 100,000 training steps. Settings are in `configs/bernoulli_dense.json`.

Run from the repository root:

```bash
python -m experiments.bernoulli_abc.bernoulli_lengthscale_abc --seed 42 --num-locations 256 --domain-stop 100 --num-blocks 2 --train-steps 100000 --checkpoint-steps 100 250 500 1000 1500 2000 5000 10000 50000 100000 --threshold-quantile 0.01 --max-proposals 10000000
```

The output directory defaults to `generated/bernoulli_abc/job_68372_dense_checkpoints/`. `visualize_abc_results.ipynb` reads this same run. `--skip-training` requires checkpoints from a completed regeneration.
