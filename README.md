# AMORe: An Agnostic Multi-Objective Framework for Recommendation

This repository contains the source code and the datasets of the paper _AMORe: An Agnostic Multi-Objective Framework for Recommendation_, accepted for publication in **ACM Transactions on Information Systems (TOIS)**.

AMORe is agnostic to the recommendation backbone. It adds beyond-accuracy objectives to the training of a recommender as differentiable approximations of top-k metrics (e.g., nDCG and APLT), which are then combined with the backbone loss.

## Requirements
We implemented and tested the code with Python `3.8.10`, `PyTorch==2.0.1` and CUDA `11.7`. The graph-based backbones (NGCF, LightGCN, MixRec) require `PyTorch Geometric`, whose packages are listed in `requirements.txt` together with the other dependencies. You can create the virtual environment as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install git+https://github.com/EdinburghNLP/torch-adaptive-imle.git   # differentiable ranking (AIMLE)
$ pip install cvxpy                                                        # EPO weighting
```

The CPFair baseline additionally requires [Gurobi](https://www.gurobi.com/).

## Data
The folder `data` contains the pre-split datasets (training, validation and test sets). Select one with the `data` key of the configuration file:

| Folder / `data` value | Dataset |
|---|---|
| `amazon_baby` | Amazon Baby |
| `amazon_book` | Amazon Book |
| `amazon_music` | Amazon Music |
| `facebook_books` | Facebook Books |
| `ml-1m` | MovieLens 1M |

## Training
All the backbones and multi-objective methods are trained through `main_unified.py`:

```
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main_unified.py --config [CONFIGURATION_FILE_NAME]
```

`CUBLAS_WORKSPACE_CONFIG` is required because the code enables PyTorch deterministic algorithms. The configuration file is looked up in the folder `config_files`. Optional arguments:

| Argument | Description |
|---|---|
| `--start i`, `--end j` | run only the experiments `i..j` of the hyperparameter grid (1-based) |
| `--reverse` | run the experiments of the grid in reverse order |
| `--track_surrogates` | AMORe only: log, at each epoch, the differentiable approximation of each objective against its exact value |

### Configuration files
A configuration file has two sections. `setting` defines the experiment. `hyperparameters` defines a grid: every key is a list of values, and one experiment is run for every combination.

```yaml
setting:
  data: facebook_books        # dataset folder
  gpu_id: 0
  baseline: BPRMF             # backbone
  wrapper: AMORE_SCALE        # multi-objective method (None for the vanilla backbone)
  epochs: 500
  validation_rate: 10         # evaluate on the validation set every N epochs
  validation_metric: ndcg@20  # ndcg@k, recall@k, precision@k, map@k or sentropy@k
  batch_size: 2048
hyperparameters:
  dim: [64]
  lr: [0.005, 0.001, 0.0005]
  l_2: [0.01, 0.005, 0.001]
  mode: [rpm]                               # objectives to optimize (see below)
  atk: [{atk_cons: 20, atk_prov: 20}]       # cutoffs of the differentiable metrics
  ranker: [AIMLE]
  scale: [0.25, 0.5, 0.75, 0.95]            # weight of the backbone loss
```

**Backbones** (`baseline`) and their specific hyperparameters:

| `baseline` | Hyperparameters |
|---|---|
| `BPRMF` | `dim`, `lr`, `l_2` |
| `NGCF` | `dim`, `lr`, `l_2`, `layers`, `message_dropout`, `node_dropout`, `normalize` |
| `LightGCN` | `dim`, `lr`, `l_2`, `layers`, `normalize` |
| `MixRec` | `dim`, `lr`, `l_2`, `layers`, `ssl_lambda`, `mix_alpha`, `temperature`, `patience` |

**Multi-objective methods** (`wrapper`):

| `wrapper` | Method | `mode` letters | Specific hyperparameters |
|---|---|---|---|
| `None` | vanilla backbone | `r` | `scale` (use 1) |
| `AMORE_SCALE` | AMORe with fixed weights (the proposed version) | `r m p d n e s` | `atk`, `ranker: AIMLE`, `scale` |
| `AMORE_MGDA`, `AMORE_EPO` | AMORe with dynamic weights computed by MGDA / EPO (ablation) | `r m p d n e s` | `atk`, `ranker: AIMLE`, `g_n` (MGDA), `scale` (EPO) |
| `AMORE_ABL`, `AMORE_ABL_WOS`, `AMORE_ABL_WOZ` | AMORe without loss normalization / without sigmoid / without z-score (ablation) | `r m p d n e s` | as `AMORE_SCALE` |
| `multifr` | MultiFR (MGDA weighting) | `r u i` | `gamma`, `temp`, `g_n`, `ranker: base` |
| `ADA2FAIR` | Ada2Fair | `r u p` | `weight_lr`, `weight_epochs`, `topk`, `provider_eta`, `alpha`, `delta`, `encoder_layers`, `decoder_layers_pfair`, `decoder_layers_ufair`, `dropout_prob`, `encoder_activation` |

With fixed weights, the backbone loss is weighted by `scale` and **each** additional AMORe objective by `1 - scale`.

**AMORe objectives** (letters of `mode`). The cutoffs are defined inside `atk`:

| Letter | Objective | Cutoff in `atk` | Other hyperparameters |
|---|---|---|---|
| `r` | backbone recommendation loss | | |
| `m` | nDCG (consumer side) | `atk_cons` (or `atk_con`) | |
| `p` | APLT, exposure of long-tail items (provider side) | `atk_prov` | |
| `d` | embedding-based intra-list diversity | `atk_div` | `item_feature_path`: `.npz` with the `item_emb` of a trained vanilla backbone (see [Outputs](#outputs)) |
| `n` | popularity-based novelty | `atk_nov` | |
| `e` | Shannon entropy of the item exposure | `atk_ent` | `entropy_mask_train` (default `True`: mask training items, as at evaluation) |
| `s` | merge all the selected objectives into a single one | | |

For MultiFR, `u` and `i` are the user-side and item-side fairness objectives. For Ada2Fair, `u` and `p` are the user-side and provider-side fairness objectives.

**Early stopping** is enabled for any backbone when the experiment defines `patience`, i.e., the number of validations without improvement before stopping.

The files in `config_files` follow the scheme `<dataset>_<backbone>_<wrapper>[_<objectives>].yml`, e.g.:
- `*_none.yml`: vanilla backbone;
- `*_AMORE_SCALE_cp.yml`, `*_AMORE_MGDA_cp.yml`, `*_AMORE_EPO_cp.yml`: AMORe with nDCG and APLT (consumer and provider side);
- `*_AMORE_SCALE_se.yml`, `*_AMORE_SCALE_d.yml`, `*_AMORE_SCALE_nd.yml`: AMORe with Shannon entropy, diversity, novelty and diversity;
- `*_multifr_cp.yml`, `*_multifr_p.yml`: MultiFR with user- and item-side fairness, or item-side fairness only;
- `*_ADA2FAIR.yml`: Ada2Fair.

### Outputs
For each experiment, identified by its hyperparameters, the following files are written:

| Path | Content |
|---|---|
| `results/<data>/recs/<id>_it=<epoch>_recs.tsv` | top-50 recommendation lists at each validation, in the Elliot format (`user`, `item`, `score`) |
| `results/<data>/performance/<setting>_validation.pkl` | validation metric of every experiment and validation epoch, used to select the best epoch |
| `results/<data>/losses/<id>_loss.pkl` | value of every loss at every batch |
| `results/<data>/parameters/<id>_params.txt` | hyperparameters of the experiment |
| `results/<data>/surrogate_history/<id>_surrogate.pkl` | with `--track_surrogates`: approximated vs. exact objectives |
| `arrays/<data>/<backbone>_None_<id>.npz` | vanilla backbones only: user/item embeddings (`user_emb`, `item_emb`) and user-item score matrix (`score_matrix`) of the best model on the validation set |

The `.npz` files are the input of CPFair and of the diversity objective (`item_feature_path`).

## Baselines
- **MultiFR** and **Ada2Fair** are trained through `main_unified.py` (see above).
- **CPFair** can be executed through the `main_cpfair.py` script. It re-ranks the score matrix predicted by a backbone, loaded from the `.npz` files in the `arrays` folder produced by the vanilla backbone runs:
  ```
  $ python3 -u main_cpfair.py
  ```
  Please note that this baseline requires the Gurobi package.

## Evaluation
To evaluate the models, we relied on the public and open-source framework [Elliot](https://github.com/sisinflab/elliot). Starting from the recommendation lists saved in `results/<data>/recs`, you can compute the metrics discussed in the paper. Please refer to the Elliot [documentation](https://elliot.readthedocs.io/en/latest/) for further details on how to compute the metrics.

## Legacy scripts
`main.py`, `main_opt.py`, `main_ada.py`, `main_tot.py`, `amore_with_shannon_entropy.py` and `amore_with_diversity_novelty.py` are the previous training scripts, kept for reference. Each of them supports only a subset of the backbones and methods, and `main_unified.py` replaces all of them.

## Citation
If you use this code, please cite our paper. The BibTeX entry will be added upon publication.
