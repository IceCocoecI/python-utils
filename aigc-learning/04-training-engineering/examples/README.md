# Module 04 Examples

These examples use a tiny synthetic classification task, so they run on CPU and
do not download datasets.

Run from this directory:

```bash
conda run -n aigc python wandb_train.py --epochs 1
conda run -n aigc python hydra_train.py --epochs 1
conda run -n aigc python optuna_search.py --trials 2 --epochs 1
conda run -n aigc python reproducible_train.py --epochs 1
```

Optional packages unlock the full third-party workflows:

```bash
pip install wandb hydra-core omegaconf optuna
```

After installing them:

```bash
WANDB_MODE=offline python wandb_train.py --use-wandb --epochs 2
python hydra_train.py training.epochs=2 optimizer.lr=0.001 model=wide
python optuna_search.py --trials 8 --epochs 2
```

Generated outputs are written under `runs/` or `outputs/` by default.
