# EFDM: point-set diffusion with variable cardinality

This repository contains the training and unconditional sampling implementations for EFDM, its soft-count variant, and the FDDM, IDM, TDDM, TDDMv, and PSD comparison methods in the manuscript. It includes the bent two-Gaussian benchmark so that the two-dimensional workflow can be run without obtaining an external dataset. The reported QM9 EFDM EMA weights are included for sampling and evaluation; QM9 records are needed only to retrain the model. Earthquake and trip records must be supplied separately.

## Contents

- `run_train.py`: train a method from a YAML configuration.
- `sample.py`: generate point sets from a trained checkpoint.
- `evaluate.py`: compute the reported metrics from saved samples.
- `methods/`, `networks/`, `trainer/`, `utils/`: method and network implementations with their shared dependencies.
- `metrics/`: feature/count Wasserstein distances, the two reported sliced-Wasserstein MMDs, JMMD, ICMMD, and QM9 chemistry scores.
- `model/qm9_efdm_epoch8000_ema.pth`: the reported final QM9 EFDM EMA weights, without optimizer or training state.
- `data/bent/generate_bent.py`: deterministic bent benchmark generator.
- `data/bent/two_gaussian_train_val_data.pt` and `data/bent/two_gaussian_test_data.pt`: the fixed splits used for the bent experiment.

Run commands from this directory. Python 3.10 or newer is recommended. Install the packages in `requirements.txt` in an environment with a PyTorch build suitable for your machine.

```bash
python -m pip install -r requirements.txt
```

## Bent two-Gaussian benchmark

Each point set draws independent `a,b ~ Uniform(0,1)`, sets `n = round(30 + 40(a+b))` and `d = 0.5 + 1.5a`, then draws each point from the equally weighted Gaussian components centered at `(−d/2, 0)` and `(d/2, 0)`, with pre-bend standard deviation `0.2`. The observed coordinates apply `(x,y) -> (x,y+x²)`. The generator uses independent NumPy PCG64 seeds 20260908, 20260909, and 20260910 for 500 training, 200 validation, and 2000 test sets. Points are stored as `float32` tensors with at most 110 rows per set and `+inf` padding. Counts are `int64`.

The included files can be verified against regeneration, or generated into a new directory:

```bash
python data/bent/generate_bent.py --verify-dir data/bent
python data/bent/generate_bent.py --output-dir bent_generated
```

The trainer accepts a YAML file. The following is the bent EFDM configuration from the paper; save it as `bent.yaml` in this directory:

```yaml
method_name: existence
dataset_type: synthetic
synthetic_data_path: data/bent/two_gaussian_train_val_data.pt
point_dim: 2
K_max: 110
embedding_dim: 128
dim_feedforward: 256
nheads: 4
num_layers: 4
exist_embedder_type: MLP
existence_net_arch: legacy
time_embedder_legacy: false
existence_eps: 0.003
weight_pos_by_exist: true
pad_mode: duplicate
existence_prepad_to_kmax: false
shuffle_points_each_epoch: true
use_dataloader: false
T: 1000
batch_size: 64
lr: 0.0001
weight_decay: 0.0
seed: 2025
num_epochs: 60000
ema_decay: 0.999
use_validation: true
validation_seed: 123
eval_interval: 100
ckpt_interval_epochs: 100
save_last_every_epochs: 100
```

```bash
python run_train.py --method_name existence --config bent.yaml --run_dir outputs/bent
python sample.py --config bent.yaml --checkpoint outputs/bent/best_model_ema.pth --weights ema --pc-mode none --num-samples 1024 --batch-size 64 --seed 42 --output outputs/bent/samples.pt
python evaluate.py --dataset bent --samples outputs/bent/samples.pt --output outputs/bent/metrics.json
```

For the other bent methods, make a copy of `bent.yaml` and replace or add the following fields. Shared fields such as `point_dim`, `embedding_dim`, `dim_feedforward`, `num_layers`, `nheads`, `seed`, `num_epochs`, `ema_decay`, and checkpoint intervals stay as above. EFDM-specific fields in the base example are ignored by the other methods.

| Paper method | `method_name` | Replace or add in YAML |
| --- | --- | --- |
| EFDM with soft-count network | `existence` | `existence_net_arch: soft_count` |
| FDDM | `mask` | `mask_net_arch: legacy`; `n_distribution_components: 5` |
| IDM | `nogroup` | `activation: relu`; `n_distribution_components: 5`; `batch_size: 4096` |
| TDDM | `jump` | `jump_net_arch: fused`; `hidden_dim: 128`; `max_n: 110`; `cutoff_ratio: 0.1`; `gamma_jump_rate: 0.27`; `lambda_adaptive: false` |
| TDDMv | `jump` | TDDM settings with `lambda_adaptive: true` |
| PSD | `psd` | `max_num_points: 256`; `num_mixture_components: 16`; `alpha_schedule: cosine`; `bce_weight: 1.0`; `nll_weight: 1.0`; `domain_margin: 0.05`; `T: 100`; `lr: 0.001`; `weight_decay: 0.0001`; `grad_clip: 2.0` |

`run_train.py --method_name` and `sample.py --method_name` can select a method without changing `method_name` in YAML. The additional aliases `existence_soft_count` and `jump_adaptive` also set their corresponding variant flags. Keep the method-specific YAML fields when using an alias.

`batch_size: 4096` for IDM counts individual points; keep `use_dataloader: false` for that method.

```bash
python run_train.py --method_name mask --config bent_mask.yaml --run_dir outputs/bent_mask
python sample.py --method_name mask --config bent_mask.yaml --checkpoint outputs/bent_mask/best_model_ema.pth --num-samples 1024 --seed 42 --output outputs/bent_mask/samples.pt
```

The trainer evaluates EMA weights with validation seed 123 every 100 epochs and writes the lowest-loss weights to `best_model_ema.pth`. For the reported bent experiment, train for 60,000 epochs and sample 1,024 nonempty sets with seed 42. `samples.pt` stores `samples` and per-batch `sampling_stats`; EFDM also stores `existence_prob` and `keep`. Baseline samples have variable cardinality and are padded to the maximum width in the output file (`+inf` for PSD, `-inf` for the other baselines). FDDM and IDM fit their set-size prior from the training counts before sampling. The test split is used for evaluation only.

## Evaluation

`evaluate.py` computes the ten spatial metrics reported for Bent, Trip, and Earthquake: Count W1; W1 for per-set mean, variance, skewness, kurtosis, and mean nearest-neighbor distance; the original PSD-style sliced-Wasserstein MMD; train-distance-scale MMD; signed JMMD²; and ICMMD². It also reports bandwidth sensitivity and ICMMD coverage. It uses every generated set and every test set. Bent evaluation reads the included splits; the full MMD pair grids are computationally expensive, so a GPU is recommended. Chamfer distance and the older Chamfer-kernel MMD are not included.

For Trip and Earthquake, supply the paper's preprocessed split and train-only normalization statistics explicitly:

```bash
python evaluate.py --dataset trip --samples outputs/trip/samples.pt --train-val path/to/train_val.pt --test path/to/test.pt --stats path/to/stats.pt --output outputs/trip/metrics.json
python evaluate.py --dataset earthquake --samples outputs/earthquake/samples.pt --train-val path/to/train_val.pt --test path/to/test.pt --stats path/to/stats.pt --output outputs/earthquake/metrics.json
```

The statistics file must contain `mean`, `std`, and `space_bound` for Trip or `space_bound_lonlat` for Earthquake. Trip JMMD/ICMMD use the processed z-score coordinates; Bent and Earthquake use the bounded metric coordinates. MMD uses the bounded metric coordinates for all three datasets. The original PSD-style MMD bandwidth comes from the generated and test distance grids; the train-scale MMD and JMMD/ICMMD bandwidths come only from training sets. The spatial tables use 4,096 accepted nonempty generated sets per method and seed; Bent uses 1,024. `evaluate.py --dataset qm9 --samples ...` computes atom stability, molecule stability, and RDKit validity from an EFDM sample file with `keep` masks.

## Other datasets

For two-dimensional data, the loader expects a PyTorch tuple `(train_data, train_counts, val_data, val_counts)`. Each data tensor has shape `[number_of_sets, padded_points, 2]`, with valid points first and remaining rows set to `+inf`. The corresponding counts tensor has one integer per set. Set `data_path` in the YAML file to the experiment's split file; the training CLI requires this path for Trip and Earthquake. Start from the Bent examples above, set `point_dim: 2`, `embedding_dim: 128`, `dim_feedforward: 256`, `num_layers: 8`, `nheads: 4`, `num_epochs: 60000`, `eval_interval: 300`, and the dataset-specific values below. IDM uses an eight-layer point-wise MLP instead of attention. Use split/training seeds 9012835, 2193852, and 6928301, one run for each seed. All methods use Adam and the lowest EMA validation loss among the 300-epoch evaluations selects the checkpoint.

| Dataset | Dataset-specific YAML values |
| --- | --- |
| Trip | `dataset_type: trip`; `data_path: path/to/seed{seed}/train_val.pt`; EFDM `K_max: 400`; PSD `max_num_points: 400`; `T: 1000`; `ema_rampup_ratio: 0.2` |
| Earthquake | `dataset_type: earthquake_psd`; `data_path: path/to/psd_split_seed{seed}_train_val_data.pt`; EFDM `K_max: 560`; PSD `max_num_points: 560`; `T: 1000`; `ema_decay: 0.9999` |

| Method | Batch size | Learning rate | Weight decay | Trip EMA half-life |
| --- | ---: | ---: | ---: | ---: |
| EFDM | 64 sets | `1e-4` | 0 | 100 epochs |
| FDDM | 64 sets | `1e-4` | 0 | 500 epochs |
| TDDM / TDDMv | 32 sets | `1e-4` | 0 | 500 epochs |
| IDM | 2,048 points | `1e-4` | 0 | 100 epochs |
| PSD | 128 sets | `1e-3` | `1e-4` | 100 epochs |

Set `ema_halflife_epochs` to the Trip value in the table. For Earthquake, use `ema_decay: 0.9999` for every method. PSD uses `T: 100` and `grad_clip: 2.0`; the other spatial methods use `T: 1000`. Give TDDM/TDDMv a `max_n` and PSD a `max_num_points` large enough for their dataset's cardinality range. Draw 4,096 nonempty sets per method and seed; the Trip paper uses sampling batches of 32.

The paper's Trip p10 files have the per-seed names `seed{seed}/train_val.pt`, `seed{seed}/test.pt`, and `seed{seed}/stats.pt`. The Earthquake random-split files are named `psd_split_seed{seed}_train_val_data.pt`, `psd_split_seed{seed}_test_data.pt`, and `psd_split_seed{seed}_stats.pt`. All three files for one seed must come from the same preprocessing run. Trip uses 268/89/91 train/validation/test sets per seed; Earthquake uses 900/300/300. The paper's Trip EMA selection assigned 100-epoch half-life to EFDM, its soft-count variant, IDM, and PSD, and 500-epoch half-life to FDDM and both TDDM variants, all with ramp-up ratio 0.2. The compact trainer accepts these selected settings but does not repeat the pilot selection.

For QM9 EFDM, the loader expects `data/qm9/train.npz` and `data/qm9/valid.npz` with `positions [N,29,3]`, `charges [N,29]` (atomic numbers, zero for padding), and `num_atoms [N]`. The model uses 29 slots, 3 coordinate channels and 5 atom-type channels, a 9-layer EGNN with hidden dimension 256, and a 4-layer Transformer with dimension 256 and eight heads. Training uses batch size 64, Adam learning rate `3e-5`, a 320,000-molecule warm-up, and an EMA half-life of 500,000 molecules with ramp-up ratio 0.05. Its training schedule is continuous VP-SDE with `beta(t)=0.1+19.9t`; the reported molecule sampling uses `pc_mode=510p485c`, corrector SNR 0.3, and no noise in the final predictor update. The reported result evaluates the final epoch-8000 EMA checkpoint once on 10,000 molecules.

Save this QM9 training example as `qm9.yaml` in this directory:

```yaml
method_name: existence
dataset_type: molecule
qm9_data_dir: data/qm9
point_dim: 8
num_atom_types: 5
K_max: 29
existence_net_arch: legacy
existence_eps: 0.003
weight_pos_by_exist: true
type_loss_weight: 1.0
pad_mode: duplicate
shuffle_points_each_epoch: true
existence_prepad_to_kmax: false
embedding_dim: 256
dim_feedforward: 256
nheads: 8
num_layers: 4
exist_embedder_type: MLP
use_p_exist_in_fuse: true
p_exist_embed_dim: 256
p_exist_time_aware: false
egnn_p_exist_input: false
egnn_feat_gate: false
use_egnn_backbone: true
egnn_hidden_nf: 256
egnn_n_layers: 9
egnn_inv_sublayers: 1
egnn_attention: true
egnn_tanh: true
egnn_norm_constant: 1
egnn_normalization_factor: 1
egnn_aggregation_method: sum
egnn_embedding_out_mode: 1
molecule_com0: true
sample_com0: false
project_pos_eps_com0: true
egnn_center_coords: true
egnn_pos_norm: 1.0
egnn_coord_clip: 50.0
T: 1000
noise_schedule: vp_sde
vp_sde_beta_min: 0.1
vp_sde_beta_max: 20.0
vp_sde_min_t: 0.001
vp_sde_no_noise_final_step: true
seed: 2025
batch_size: 64
lr: 0.00003
lr_warmup_kimg: 320
weight_decay: 0.0
grad_clip: 1.0
ema_halflife_kimg: 500
ema_rampup_ratio: 0.05
num_epochs: 8000
use_validation: false
ckpt_interval_epochs: 100
save_last_every_epochs: 100
```

```bash
python run_train.py --config qm9.yaml --run_dir outputs/qm9
```

The reported sampling and evaluation commands use the included final epoch-8000 EMA weights. The checkpoint SHA-256 is `aeca8246d40ead7debdd42c4d1f0b5f0fbf534c505b62e075694c2b37a445f6e`.

```bash
python sample.py --config qm9.yaml --checkpoint model/qm9_efdm_epoch8000_ema.pth --weights ema --pc-mode 510p485c --vp-no-noise-final-step --num-samples 10000 --batch-size 64 --seed 10000 --output outputs/qm9/samples.pt
python evaluate.py --dataset qm9 --samples outputs/qm9/samples.pt --output outputs/qm9/metrics.json
```

## Reproduction limits

The included Bent splits and generator support the complete two-dimensional training, sampling, and metric workflow. Trip, Earthquake, and QM9 raw data are omitted. Reproducing their training requires the exact paper split and preprocessing; an arbitrary file with the right tensor shape does not reproduce the tables. The included QM9 checkpoint permits sampling and chemistry-metric evaluation without the QM9 training data. Trip's reported EMA rule was selected by a separate 6,000-epoch pilot with three validation seeds, while this compact trainer tracks one configured EMA and one validation seed. The QM9 result uses the final EMA checkpoint after 8,000 epochs; no checkpoint screening is needed.

## Checkpoints and scope

The training entry point writes checkpoints to the directory passed with `--run_dir`. `sample.py` accepts a weight-only checkpoint or a resumable `last.pth`. This repository includes only the reported QM9 EMA weights, so it cannot resume that training run. A fixed seed controls generation, but exact floating-point agreement can still depend on the PyTorch version and hardware.

The source code retains the method and network components needed for the manuscript's training and sampling.

## License

The code is released under the MIT license in `LICENSE`. The adapted EGNN layers retain their upstream copyright and MIT license in `THIRD_PARTY_NOTICES.md`.
