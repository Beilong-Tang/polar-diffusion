# polar-diffusion

Code for polar diffusion models. A noisy sample $y_t$ is written in polar form
$y_t = r_t x_t$, with radius $r_t = \lVert y_t \rVert$ and direction $x_t$ on the unit sphere.
The network sees the direction as its image input and the radius through an
embedding. It predicts the noise $\epsilon$ split into two parts:

- a **tangent** part $\epsilon_x = \epsilon - \langle x_t, \epsilon\rangle x_t$
- a **radial** part $\epsilon_r = \langle x_t, \epsilon\rangle$

Training minimizes

$$\mathcal{L} = w_x \,\tfrac{1}{n}\lVert \epsilon_x - \hat\epsilon_x \rVert^2 + \tfrac{w_r}{n}\,(\epsilon_r - \hat\epsilon_r)^2 .$$

Samples come from the reverse SDE in polar coordinates: $\rho = \log r$ moves on
the real line and $x$ moves on the sphere through the exponential map. The model
can also be sampled with standard DDPM/DDIM steps, using the recombined
prediction $\hat\epsilon = \hat\epsilon_r x_t + \hat\epsilon_x$.

The same polar core (`polar_diffusion/polar.py`) is used by two backbones:

| experiment | backbone | adapted from |
|---|---|---|
| CIFAR-10 32×32 | `polar_diffusion.ddpm_torch` | [tqch/ddpm-torch](https://github.com/tqch/ddpm-torch) |
| ImageNet-64 (unconditional) | `polar_diffusion.improved_diffusion` | [openai/improved-diffusion](https://github.com/openai/improved-diffusion) |

## Installation

```bash
pip install -e .
pip install -e ".[webdataset]"    # optional: stream ImageNet from .tar shards
pip install -e ".[monitor]"       # optional: GPU utilization in the training logs
```

## Repository layout

```
polar_diffusion/
  geometry.py            normalize / tangent projection / exponential map on the sphere
  polar.py               polar loss, polar reverse-SDE step, radius embedding & read-out, radius statistics
  utils.py               logging, torchrun setup, resumable sample indexing
  ddpm_torch/            CIFAR-10 backbone (diffusion, UNet, DDIM, datasets, trainer)
  improved_diffusion/    ImageNet-64 backbone (diffusion, respacing, UNet, training loop)
scripts/
  cifar10/               train.py, generate.py
  imagenet64/            radius_stats.py, train.py, sample.py
configs/
  cifar10.json           CIFAR-10 model / diffusion / polar / training config
  imagenet64_radius.json radius statistics of ImageNet-64
shell/                   end-to-end pipelines: cifar10.sh, imagenet64.sh
```

## CIFAR-10

The whole pipeline (train, then generate 50k samples) is in
`bash shell/cifar10.sh`. The individual steps:

```bash
# train (4 GPUs); the dataset is downloaded to --root if missing
python scripts/cifar10/train.py --config-path configs/cifar10.json --exp-name cifar10_polar \
    --root ~/datasets --distributed --rigid-launch --num-gpus 4 --resume

# generate 50k samples with the polar sampler (all visible GPUs)
python scripts/cifar10/generate.py --config-path configs/cifar10.json \
    --chkpt-path chkpts/cifar10_polar/cifar10_polar_2040.pt --save-dir output/cifar10_polar_2040
```

The `"polar"` block of `configs/cifar10.json` holds the polar settings. Each one can be
overridden on the command line (`--wr 0.01`, `--field rho`, `--std-r false`, …).

| key | meaning |
|---|---|
| `wx`, `wr` | weights of the tangent and radial loss terms |
| `std_r` | standardize the radius input with training-set statistics (cached as `std.json` in the checkpoint directory) |
| `field` | radius input: `r` or `rho = log r` |
| `mlp_pool_hidden` | hidden layer in the radial read-out head |
| `condition` | `concat` or `add` the radius embedding to the time embedding |
| `proj_inf` | project the tangent prediction onto the tangent space when sampling |
| `scale_x`, `input_field` | ablations of the network input (scaled direction, or the raw sample `y`) |

Sampler options for `generate.py`:

- `--sample-method polar|ddpm`
- `--polar-beta-threshold β`: take DDPM steps wherever $\beta_t > β$, i.e. a hybrid sampler
- `--use-ddim --subseq-size 50`

## ImageNet-64

The whole pipeline is in `bash shell/imagenet64.sh`; set the data paths at the top first. The individual steps:

```bash
# 1. radius statistics (configs/imagenet64_radius.json ships with the repo)
python scripts/imagenet64/radius_stats.py --data /path/to/imagenet64.npy --out configs/imagenet64_radius.json

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --radius_stats configs/imagenet64_radius.json"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# 2. train; --batch_size is per GPU, and the run resumes from the latest checkpoint in --logdir
torchrun --standalone --nproc_per_node=1 scripts/imagenet64/train.py --data_dir /path/to/imagenet64.npy \
    --logdir logs/imagenet64_polar $MODEL_FLAGS $DIFFUSION_FLAGS --lr 1e-4 --batch_size 192 --total_step 200000

# 3. sample 10k images; --batch_size is the total over GPUs
torchrun --standalone --nproc_per_node=8 scripts/imagenet64/sample.py $MODEL_FLAGS $DIFFUSION_FLAGS \
    --model_path logs/imagenet64_polar/ema_0.9999_200000.pt --outdir logs/imagenet64_polar/eval_polar \
    --num_samples 10000 --batch_size 64 --sample_method polar
```

`--data_dir` accepts a folder of images, a `.npy` array of uint8 images `[N, 64, 64, 3]`, or a
WebDataset shard pattern such as `"/data/imagenet64-{0001..1282}.tar"`. Polar flags:
`--wx`, `--wr`, `--radius_field`, `--radius_stats`, `--condition`, `--mlp_pool_hidden`,
`--proj_inf`. Sampler flags: `--sample_method polar|ddpm`, `--polar_beta_threshold`,
`--use_ddim True`, `--timestep_respacing 250`.

Both sampling scripts resume: re-running the same command generates only the
images that are missing.

## Acknowledgements

The backbones are adapted from [ddpm-torch](https://github.com/tqch/ddpm-torch) (MIT, © 2022 Tianqi Chen)
and [improved-diffusion](https://github.com/openai/improved-diffusion) (MIT, © 2021 OpenAI). Their licenses
are kept in the corresponding subpackages.
