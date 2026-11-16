# Diffusion models from scratch

A class-conditional DDPM written in plain PyTorch, built in small phases. The goal is to train it on CIFAR-10 and show the sampling process as a live demo on [armansoylu.me](https://armansoylu.me).

**This is work in progress.** The training scaffold works. The model code is still a draft.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Training scaffold: multi-GPU, mixed precision, checkpoint and resume, time budget | Works, exercised with a dummy model and random data |
| 1 | Toy DDPM on 2D two-moons data with classifier-free guidance (`toy.py`) | In progress: data, noise schedule and forward process are written, the model and the training loop are not finished |
| 2 | Class-conditional U-Net for 32×32 images (`unet.py`) | Draft, not runnable yet |
| 3 | Training on CIFAR-10, sampling, FID | Not started |

Each phase is meant to be verified on its own before the next one depends on it.

## Phase 0: the training scaffold

`train.py` is the loop the real model will be dropped into. It was written first because the training target is Kaggle GPU sessions (T4, 12-hour limit), where a run has to survive being cut off.

- **One code path for one or many GPUs.** Runs as a single process, or under `torchrun` with `DistributedDataParallel`. Without a GPU it falls back to CPU and the `gloo` backend.
- **Mixed precision.** fp16 autocast with a gradient scaler on CUDA, switched off on CPU.
- **Atomic checkpoints.** A checkpoint is written to a temporary file and then renamed, so an interrupted write never leaves a broken file. The two most recent ones are kept as `last.pt` and `prev.pt`.
- **Automatic resume.** A run continues from `last.pt` in its output folder, or from the file given with `--resume`. Model, optimizer and scaler state are restored.
- **Time budget.** `--max-hours` stops the run before the session limit. All ranks agree on the stop, then a final checkpoint is saved.
- **Plain logging.** Step, loss and iterations per second go to `log.jsonl`.

### Run it

```bash
pip install -r requirements.txt

# single process (CPU or one GPU)
python train.py --config configs/dummy_config.yaml --out run

# two GPUs
torchrun --nproc_per_node=2 train.py --config configs/dummy_config.yaml --out run

# continue the same run for more steps
python train.py --config configs/dummy_config.yaml --out run --total-steps 600
```

`configs/dummy_config.yaml` is the smoke test for this phase: 300 steps, a checkpoint every 100. `batch_size` in the config is the global batch and is split across the GPUs.

## Planned model

These are the choices currently written into the drafts:

- U-Net with 128 base channels and channel multipliers (1, 2, 2, 2)
- Residual blocks with GroupNorm and SiLU, conditioned on a sinusoidal timestep embedding plus a class embedding
- Self-attention at 16×16 and 8×8, and in the middle block
- Linear noise schedule from 1e-4 to 0.02 over 1000 steps
- Classifier-free guidance, with the class label dropped 10% of the time during training

## Files

| File | Purpose |
|---|---|
| `train.py` | Training scaffold (phase 0) |
| `configs/dummy_config.yaml` | Smoke-test configuration |
| `toy.py` | 2D toy DDPM (phase 1, in progress) |
| `unet.py` | U-Net for CIFAR-10 (phase 2, draft) |

## Next steps

1. Finish the toy model and train it on two-moons, with plots of the samples
2. Get the U-Net to run and check its shapes with a forward pass
3. Train on CIFAR-10, add sampling, report FID and a sample grid
