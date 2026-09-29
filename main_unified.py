"""Unified training entry point.

Backbones (setting.baseline): BPRMF, LightGCN, NGCF, MixRec.
Multi-objective methods (setting.wrapper):
  None                                   -> backbone only
  AMORE_SCALE / AMORE_MGDA / AMORE_EPO   -> AMORe with static / MGDA / EPO weighting
  AMORE_ABL / AMORE_ABL_WOS / AMORE_ABL_WOZ -> AMORe ablations (static weighting)
  multifr                                -> MultiFR (MGDA weighting)
  ADA2FAIR                               -> Ada2Fair (two-stage adaptive weighting)

AMORe objectives (letters of `mode`, cutoffs inside `atk`):
  m nDCG (atk_cons/atk_con), p APLT (atk_prov), d embedding diversity ILD (atk_div, item_feature_path),
  n popularity novelty (atk_nov), e Shannon entropy of the exposure (atk_ent, entropy_mask_train),
  s merge all of them into a single objective. r is the backbone loss for every method.

Early stopping is enabled when the experiment defines `patience` (in validation steps).

Usage:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u main_unified.py --config <file.yml> [--start i] [--end j]
"""
import itertools
import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from imle.aimle import aimle
from imle.target import AdaptiveTargetDistribution
from scipy.stats import spearmanr
from torch import Tensor
from torch.nn import Sigmoid
from tqdm import tqdm

from Namespace import Namespace
from Namespace_nd import NamespaceND
from SoftRank import SmoothDCGLoss, SmoothRank
from early_stopping import EarlyStopping
from epo_lp import EPO_LP
from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k
from min_norm_solvers import MinNormSolver, gradient_normalizers
from model.ada2fair import Ada2FairModel
from model.lightgcn import LightGCNModel
from model.mf import MatrixFactorization
from model.mixrec import MixRecModel
from model.ngcf import NGCFModel
from preprocess import preprocessing
from sampler import NegSampler, negsamp_vectorized_bsearch_preverif

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ======================================================================
# Differentiable ranking (AIMLE)
# ======================================================================

def rank(seq: Tensor) -> Tensor:
    res = torch.argsort(torch.argsort(seq, dim=1, descending=True)) + 1
    return res.float()


# Adaptive Implicit MLE (https://arxiv.org/abs/2209.04862, AAAI 2023)
target_distribution = AdaptiveTargetDistribution(beta_update_step=1e-2)


@aimle(target_distribution=target_distribution)
def differentiable_ranker(weights_batch: Tensor) -> Tensor:
    return rank(weights_batch)


class AIMLE_ranking:
    def __call__(self, input: Tensor) -> Tensor:
        return differentiable_ranker(input)


# ======================================================================
# Generic utilities
# ======================================================================

def parse_args():
    parser = ArgumentParser(description="Unified framework for multi-objective recommendation")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--track_surrogates', action='store_true',
                        help='AMORe only: log approximated vs true nDCG/APLT at each epoch')
    parser.add_argument('--reverse', action='store_true',
                        help='run the experiments of the grid in reverse order')
    return parser.parse_args()


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable


def spearman_corr(x, y):
    return float(spearmanr(x, y).correlation)


def pearson_corr(x, y, eps=1e-8):
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    return float((x * y).mean() / (x.std() * y.std() + eps))


def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices
        user_neg_items.append(negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates))
    return np.asarray(user_neg_items)


def generate_pred_list(model, train_matrix, device, topk=20, batch_size=1024):
    model.eval()
    num_users = train_matrix.shape[0]
    pred_chunks = []
    with torch.no_grad():
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_user_ids = torch.arange(start, end, dtype=torch.long, device=device)
            rating_pred = model.predict(batch_user_ids)
            seen_mask = torch.as_tensor(train_matrix[start:end].toarray(), dtype=torch.bool,
                                        device=rating_pred.device)
            rating_pred = rating_pred.masked_fill(seen_mask, float('-inf'))
            _, batch_items = torch.topk(rating_pred, k=topk, dim=1)
            pred_chunks.append(batch_items.cpu().numpy())
    return np.concatenate(pred_chunks, axis=0)


def shannon_entropy_at_k(pred_list, k, num_items, eps=1e-12):
    """Normalized Shannon entropy of the item exposure in the top-k lists (as Elliot SEntropy),
    divided by log2(min(num_items, num_users * k)) to lie in [0, 1]."""
    topk = np.asarray(pred_list[:, :k], dtype=np.int64)
    num_users = topk.shape[0]
    if num_users == 0 or k <= 0:
        return 0.0
    counts = np.bincount(topk.reshape(-1), minlength=num_items)
    probs = counts[counts > 0].astype(np.float64) / float(num_users * k)
    entropy = float(-(probs * np.log2(probs + eps)).sum())
    support = min(int(num_items), num_users * k)
    return entropy / math.log2(support) if support > 1 else 0.0


def compute_metrics(test_set, pred_list, metric, num_items):
    metric, k = metric.split('@')[0], int(metric.split('@')[1])
    if metric == 'ndcg':
        return ndcg_k(test_set, pred_list, k)
    elif metric == 'recall':
        return recall_at_k(test_set, pred_list, k)
    elif metric == 'precision':
        return precision_at_k(test_set, pred_list, k)
    elif metric == 'map':
        return mapk(test_set, pred_list, k)
    elif metric in ['sentropy', 'shannon', 'entropy', 'se']:
        return shannon_entropy_at_k(pred_list, k, num_items)
    raise ValueError(f"Unknown validation metric: {metric}")


def rec_to_elliot(iter, top_ids, dataset, exp_string, data_name):
    num_users, top_k = top_ids.shape
    df = pd.DataFrame({
        'user': np.repeat(np.arange(num_users), top_k),
        'item': top_ids.flatten(),
        'rating': np.tile(np.arange(top_k, 0, -1), num_users),
    })
    df['user'] = df['user'].map(pd.Series(dataset['user_mapping_inv']))
    df['item'] = df['item'].map(pd.Series(dataset['item_mapping_inv']))
    if df['user'].isnull().any() or df['item'].isnull().any():
        raise ValueError("Failed Mapping. NaN")

    output_dir = f'results/{data_name}/recs'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/{exp_string}_it={iter}_recs.tsv', sep='\t', index=False, header=False)


# Keys excluded from the experiment identifier. Same convention as main_tot.py, so that
# the output file names stay compatible with the best_epoch_*.py scripts.
EXP_ID_EXCLUDED = ['backbone', 'mo_method', 'mode', 'device', 'every', 'metric', 'batch_size', 'n_epochs',
                   'ranker', 'atk', 'delta', 'encoder_layers', 'decoder_layers_pfair', 'decoder_layers_ufair',
                   'dropout_prob', 'encoder_activation', 'patience', 'item_feature_path']


def exp_string(i, args):
    head = '-'.join(f'{key}={value}' for key, value in vars(args).items()
                    if key in ['backbone', 'mo_method', 'mode'])
    tail = '-'.join(f'{key}={value}' for key, value in vars(args).items()
                    if key not in EXP_ID_EXCLUDED).replace('.', '$')
    return str(i) + '-' + head + '-' + tail


def exp_setting(setting):
    return '-'.join(f'{key}={value}' for key, value in vars(setting).items()
                    if key in ['backbone', 'mo_method', 'mode', 'data'])


# ======================================================================
# Backbones
# ======================================================================

def build_backbone(args, data):
    if args.backbone == 'BPRMF':
        model = MatrixFactorization(data.user_size, data.item_size, args)
    elif args.backbone == 'LightGCN':
        model = LightGCNModel(data.user_size, data.item_size, args, data.train_matrix)
    elif args.backbone == 'NGCF':
        model = NGCFModel(data.user_size, data.item_size, args, data.train_matrix)
    elif args.backbone == 'MixRec':
        model = MixRecModel(data.user_size, data.item_size, args, data.train_matrix)
    else:
        raise ValueError(f"Backbone not supported: {args.backbone}")
    return model.to(args.device)


@torch.no_grad()
def extract_embeddings(model, backbone):
    """Final user/item embeddings, consistent with what model.predict() uses."""
    if backbone == 'BPRMF':
        gu, gi = model.user_embeddings.weight, model.item_embeddings.weight
    elif backbone == 'LightGCN':
        gu, gi = model.propagate_embeddings(evaluate=True)
    elif backbone == 'NGCF':
        gu, gi = model.propagate_embeddings(model.adj)
    elif backbone == 'MixRec':
        gu, gi = model.propagate_embeddings()
        gi = gi[:-1]  # remove padding item
    else:
        raise ValueError(f"Backbone not supported: {backbone}")
    return gu.detach().cpu().numpy(), gi.detach().cpu().numpy()


def save_backbone_arrays(model, args, exp_id, data):
    os.makedirs(f'arrays/{args.data}/', exist_ok=True)
    user_emb, item_emb = extract_embeddings(model, args.backbone)
    with torch.no_grad():
        all_users = torch.arange(data.user_size, dtype=torch.long, device=args.device)
        score_matrix = model.predict(all_users).detach().cpu().numpy()
    rows, cols = data.train_matrix.nonzero()
    score_matrix[rows, cols] = -1e9
    np.savez_compressed(
        f'arrays/{args.data}/{args.backbone}_{args.mo_method}_{exp_id}.npz',
        user_emb=user_emb,
        item_emb=item_emb,
        score_matrix=score_matrix,
        user_map=np.array(data.dataset['user_mapping_inv'], dtype=object),
        item_map=np.array(data.dataset['item_mapping_inv'], dtype=object),
        metric_name=np.array([args.metric], dtype=object)
    )


# ======================================================================
# Item-side resources of the AMORe beyond-accuracy objectives
# ======================================================================

def soft_topk_mask(ranks, k):
    """Differentiable approximation of the top-k membership mask."""
    return (torch.tanh(-ranks + k) + 1.0) / 2.0


def load_fixed_item_features(path, item_size, device):
    """Fixed (L2-normalized) item representations for the embedding diversity objective, taken
    from the item_emb of a .npz saved by a vanilla backbone run. Keeping them fixed avoids a
    self-referential objective that could be optimized by moving the item embeddings."""
    arr = np.load(os.path.expanduser(str(path)), allow_pickle=True)
    if 'item_emb' not in arr.files:
        raise KeyError(f"Expected key 'item_emb' in {path}, found keys: {arr.files}")
    item_features = arr['item_emb'].astype(np.float32)
    if item_features.shape[0] != item_size:
        raise ValueError(f"item_emb has {item_features.shape[0]} rows, but the dataset has {item_size} items. "
                         f"Check that the feature file uses the same item mapping.")
    return F.normalize(torch.as_tensor(item_features, device=device), p=2, dim=1)


def build_item_novelty_scores(train_matrix, device):
    """Popularity-based novelty in [0, 1]: close to 1 for rare items, 0 for the most popular one."""
    item_popularity = np.asarray(train_matrix.getnnz(axis=0), dtype=np.float32)
    max_popularity = float(max(item_popularity.max(), 1.0))
    novelty = np.clip(1.0 - np.log1p(item_popularity) / np.log1p(max_popularity), 0.0, 1.0)
    return torch.as_tensor(novelty.astype(np.float32), device=device)


# ======================================================================
# Loss weighting strategies (shared by the multi-objective methods)
# ======================================================================

def static_weights(tasks, scale1):
    return {t: (scale1 if t == '1' else 1.0 - scale1) for t in tasks}


def task_gradients(model, optimizer, loss, t, device):
    optimizer.zero_grad()
    loss[t].backward(retain_graph=True)
    return [p.grad.detach().clone().to(device).flatten() for p in model.parameters() if p.grad is not None]


def mgda_weights(model, optimizer, loss, tasks, normalization, device):
    grads = {t: task_gradients(model, optimizer, loss, t, device) for t in tasks}
    loss_data = {t: loss[t].item() for t in tasks}
    gn = gradient_normalizers(grads, loss_data, normalization)
    for t in tasks:
        if gn[t] == 0.0:
            gn[t] += 1
        grads[t] = [g / gn[t] for g in grads[t]]
    sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
    return {t: float(sol[i]) for i, t in enumerate(tasks)}


class EPOWeighting:
    def __init__(self, model, tasks, scale1):
        _, n_params = getNumParams(model.parameters())
        self.preference = np.array([scale1] + [1 - scale1] * (len(tasks) - 1))
        self.solver = EPO_LP(m=len(tasks), n=n_params, r=self.preference)
        self.reset_counters()

    def reset_counters(self):
        self.n_linscalar_adjusts = 0
        self.descent = 0

    def __call__(self, model, optimizer, loss, tasks, device):
        values = [loss[t].data.cpu().numpy() for t in tasks]
        G = torch.stack([torch.cat(task_gradients(model, optimizer, loss, t, device)) for t in tasks])
        try:
            sol = self.solver.get_alpha(np.stack(values), G=(G @ G.T).cpu().numpy(), C=True)
            if self.solver.last_move == "dom":
                self.descent += 1
        except Exception:
            sol = None
        if sol is None:  # A patch for the issue in cvxpy
            sol = self.preference / self.preference.sum()
            self.n_linscalar_adjusts += 1
        return {t: float(len(tasks) * sol[i]) for i, t in enumerate(tasks)}


# ======================================================================
# Multi-objective methods
# ======================================================================

class Backbone:
    """Plain backbone training (wrapper: None). Also the base class of the other methods."""
    needs_sampled_items = False
    MODES = 'r'

    @classmethod
    def validate(cls, args):
        unknown = sorted(set(args.mode) - set(cls.MODES))
        if unknown:
            raise ValueError(f"mode '{args.mode}': objectives {unknown} are not supported by wrapper "
                             f"'{args.mo_method}' (allowed: '{cls.MODES}')")

    def __init__(self, args, data, cli):
        self.validate(args)
        self.args = args
        self.data = data
        self.tasks = ['1']

    def wrap(self, backbone):
        return backbone

    def setup(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

    def set_sampled_items(self, sampled_ids, labels):
        self.sampled_ids, self.labels = sampled_ids, labels

    def on_epoch_start(self, model, sampler, num_batches):
        pass

    def on_epoch_end(self, epoch):
        pass

    def zero(self):
        return torch.tensor(0.0, device=self.args.device)

    def rec_loss(self, model, batch):
        if 'r' in self.args.mode:
            return model(batch.user_id, batch.pos_id, batch.neg_id)
        return self.zero()

    def compute_losses(self, model, batch):
        return {'1': self.rec_loss(model, batch)}

    def weights(self, model, loss):
        return static_weights(self.tasks, self.args.scale1)

    def step(self, model, loss, scale, history):
        batch_loss = 0
        for t in self.tasks:
            history[f'loss_{t}'].append((loss[t].item(), loss[t].item() * scale[t]))
            batch_loss += loss[t] * scale[t]
        history['batch_loss'].append(batch_loss.item())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

    def log_weights(self, model, scale):
        print('\n'.join('\'{:s}\': {:.10f}'.format(k, scale[k]) for k in self.tasks))


class Amore(Backbone):
    """AMORe: each objective is a differentiable top-k metric in [0, 1] (utopia point 1),
    optimized through the loss sum_u normalize((1 - metric_u)^2). With 's' all of them are
    merged into a single auxiliary objective."""
    needs_sampled_items = True
    MODES = 'rmpsdne'
    # (mode letter, task id, metric name)
    OBJECTIVES = [('m', '2', 'ndcg'), ('p', '3', 'aplt'), ('d', '5', 'ild'), ('n', '6', 'novelty'),
                  ('e', '7', 'entropy')]
    NORMALIZERS = {
        'AMORE_SCALE': 'zscore_sigmoid', 'AMORE_MGDA': 'zscore_sigmoid', 'AMORE_EPO': 'zscore_sigmoid',
        'AMORE_ABL_WOS': 'zscore', 'AMORE_ABL_WOZ': 'sigmoid', 'AMORE_ABL': 'none',
    }

    @classmethod
    def validate(cls, args):
        super().validate(args)
        if args.ranker != 'AIMLE':
            raise NotImplementedError(f"AMORe supports only ranker 'AIMLE' (got '{args.ranker}')")
        for letter, key in [('m', 'atk_con'), ('p', 'atk_pro')]:
            if letter in args.mode and not hasattr(args, key):
                raise ValueError(f"AMORe with mode '{args.mode}' requires the cutoff '{key}' inside atk "
                                 f"(e.g. atk: [{{atk_cons: 20, atk_prov: 20}}])")
        if 'd' in args.mode:
            path = getattr(args, 'item_feature_path', None)
            if path is None or not os.path.exists(os.path.expanduser(str(path))):
                raise ValueError(f"mode 'd' requires item_feature_path pointing to a .npz with item_emb "
                                 f"(e.g. saved by a vanilla backbone run); got: {path}")

    def __init__(self, args, data, cli):
        super().__init__(args, data, cli)
        self.tasks = []
        if 'r' in args.mode:
            self.tasks.append('1')
        if 's' in args.mode:
            self.tasks.append('4')
        else:
            self.tasks += [t for letter, t, _ in self.OBJECTIVES if letter in args.mode]
        self.normalization = self.NORMALIZERS[args.mo_method]
        self.long_tail = data.long_tail
        self.track_surrogates = cli.track_surrogates
        self.surrogate_history = None
        if self.track_surrogates:
            fields = ['approx_mean', 'true_mean', 'gap_mean', 'abs_gap_mean', 'pearson', 'spearman']
            self.surrogate_history = {m: {f: [] for f in fields} for m in ['ndcg', 'aplt', 'ild', 'novelty']}

    def setup(self, model):
        super().setup(model)
        self.ranker = AIMLE_ranking()
        if self.args.mo_method == 'AMORE_EPO':
            self.epo = EPOWeighting(model, self.tasks, self.args.scale1)
        if 'd' in self.args.mode:
            self.item_features = load_fixed_item_features(self.args.item_feature_path, self.data.item_size,
                                                          self.args.device)
        if 'n' in self.args.mode:
            self.novelty_scores = build_item_novelty_scores(self.data.train_matrix, self.args.device)

    def cutoff(self, *names, default=20):
        for name in names:
            if hasattr(self.args, name):
                return int(getattr(self.args, name))
        return default

    def normalize(self, data):
        if self.normalization in ['zscore_sigmoid', 'zscore']:
            data = (data - torch.mean(data)) / torch.std(data)
        if self.normalization in ['zscore_sigmoid', 'sigmoid']:
            data = Sigmoid()(data)
        return data

    # ----- differentiable metrics (one value per user of the batch) -----

    def differentiable_ndcg(self, ranks, sampled_ids, labels):
        k = self.args.atk_con
        idcg = sum(1.0 / math.log(i + 2, 2) for i in range(k))
        gathered_ranks = ranks.gather(1, sampled_ids)
        dcg_num = ((torch.tanh(-gathered_ranks + k) + 1) / 2) * labels.float()
        return torch.sum(dcg_num / torch.log2(gathered_ranks + 1), dim=-1) / idcg

    def differentiable_aplt(self, ranks):
        ranks = (torch.tanh(-ranks + self.args.atk_pro) + 1) / 2
        return torch.sum(ranks[:, self.long_tail], dim=1) / self.args.atk_pro

    def differentiable_ild(self, ranks, eps=1e-8):
        """Embedding-based intra-list diversity: average pairwise (1 - cosine) / 2 among the soft top-k.
        sum_ij w_i w_j cos(i, j) = ||W F||^2, so the I x I similarity matrix is never built."""
        weights = soft_topk_mask(ranks, self.cutoff('atk_div', 'atk_pro'))
        weighted_features = weights @ self.item_features  # keep this order: it fixes the gradient summation order
        sum_w, sum_w2 = weights.sum(dim=1), torch.square(weights).sum(dim=1)
        cosine_sum = torch.square(weighted_features).sum(dim=1) - sum_w2
        avg_cosine = cosine_sum / (torch.square(sum_w) - sum_w2).clamp_min(eps)
        return torch.clamp((1.0 - avg_cosine) / 2.0, min=0.0, max=1.0)

    def differentiable_novelty(self, ranks, eps=1e-8):
        """Popularity-based novelty of the soft top-k."""
        weights = soft_topk_mask(ranks, self.cutoff('atk_nov', 'atk_pro'))
        return (weights * self.novelty_scores.view(1, -1)).sum(dim=1) / weights.sum(dim=1).clamp_min(eps)

    def differentiable_entropy(self, scores, ranks, users, eps=1e-12):
        """Per-user contribution to the normalized Shannon entropy of the soft top-k item exposure
        over the batch. By default training items are masked out (as in evaluation), which needs
        a second rank approximation."""
        if getattr(self.args, 'entropy_mask_train', True):
            seen = torch.as_tensor(self.data.train_matrix[users.cpu().numpy()].toarray(), dtype=torch.bool,
                                   device=scores.device)
            ranks = self.ranker(scores.masked_fill(seen, -1e8))
        k = self.cutoff('atk_ent', 'atk_div', 'atk_pro')
        c = soft_topk_mask(ranks, k)
        user_mass = c / (c.sum(dim=1, keepdim=True) + eps)
        p_item = user_mass.sum(dim=0) / (user_mass.size(0) + eps)
        entropy_user = (user_mass * -torch.log2(p_item + eps).unsqueeze(0)).sum(dim=1)
        max_support = min(ranks.size(1), max(2, ranks.size(0) * k))
        denom = torch.log2(torch.tensor(float(max_support), device=ranks.device))
        return torch.clamp(entropy_user / (denom + eps), min=0.0, max=1.0)

    def differentiable_metric(self, name, scores, ranks, sampled_ids, labels, users):
        if name == 'ndcg':
            return self.differentiable_ndcg(ranks, sampled_ids, labels)
        if name == 'aplt':
            return self.differentiable_aplt(ranks)
        if name == 'ild':
            return self.differentiable_ild(ranks)
        if name == 'novelty':
            return self.differentiable_novelty(ranks)
        return self.differentiable_entropy(scores, ranks, users)

    # ----- exact metrics, only used to monitor the surrogates (--track_surrogates) -----

    def true_metric(self, name, scores, sampled_ids, labels):
        if name == 'ndcg':
            k = self.args.atk_con
            idcg = sum(1.0 / math.log(i + 2, 2) for i in range(k))
            gathered_ranks = rank(scores).gather(1, sampled_ids).float()
            topk_mask = (gathered_ranks <= k).float()
            return torch.sum(topk_mask * labels.float() / torch.log2(gathered_ranks + 1), dim=-1) / idcg
        if name == 'aplt':
            topk_items = torch.topk(scores, k=self.args.atk_pro, dim=1).indices
            long_tail = torch.as_tensor(self.long_tail, device=scores.device)
            return torch.isin(topk_items, long_tail).float().sum(dim=1) / self.args.atk_pro
        if name == 'ild':
            k = self.cutoff('atk_div', 'atk_pro')
            feats = self.item_features[torch.topk(scores, k=k, dim=1).indices]
            cosine_sum = torch.matmul(feats, feats.transpose(1, 2)).sum(dim=(1, 2)) - k
            return torch.clamp((1.0 - cosine_sum / max(k * (k - 1.0), 1e-8)) / 2.0, min=0.0, max=1.0)
        if name == 'novelty':
            k = self.cutoff('atk_nov', 'atk_pro')
            return self.novelty_scores[torch.topk(scores, k=k, dim=1).indices].mean(dim=1)
        return None  # entropy is a global metric: no per-user exact value

    def on_epoch_start(self, model, sampler, num_batches):
        if self.track_surrogates:
            self.epoch_values = {m: {'approx': [], 'true': []} for m in self.surrogate_history}
        if self.args.mo_method == 'AMORE_EPO':
            self.epo.reset_counters()

    def compute_losses(self, model, batch):
        loss = {'1': self.rec_loss(model, batch)}
        u = batch.unique_u
        scores_all = model.predict(u)  # [B, I], only the users of the batch
        ranks = self.ranker(scores_all)
        sampled_ids, labels = self.sampled_ids[u], self.labels[u]

        metrics = {}
        for letter, t, name in self.OBJECTIVES:
            if letter in self.args.mode:
                metrics[name] = self.differentiable_metric(name, scores_all, ranks, sampled_ids, labels, u)
                loss[t] = self.normalize(torch.square(1 - metrics[name])).sum()
            else:
                loss[t] = self.zero()

        aux_tasks = [t for _, t, _ in self.OBJECTIVES]
        if 's' in self.args.mode:
            loss['4'] = sum(loss[t] for t in aux_tasks) / len(u)
            for t in aux_tasks:
                loss[t] = self.zero()
        else:
            for t in aux_tasks:
                loss[t] = loss[t] / len(u)

        if self.track_surrogates:
            with torch.no_grad():
                for name, values in metrics.items():
                    true = self.true_metric(name, scores_all, sampled_ids, labels)
                    if true is not None:
                        self.epoch_values[name]['approx'].extend(values.detach().cpu().tolist())
                        self.epoch_values[name]['true'].extend(true.cpu().tolist())
        return loss

    def weights(self, model, loss):
        if self.args.mo_method == 'AMORE_MGDA':
            return mgda_weights(model, self.optimizer, loss, self.tasks, self.args.type, self.args.device)
        if self.args.mo_method == 'AMORE_EPO':
            return self.epo(model, self.optimizer, loss, self.tasks, self.args.device)
        return static_weights(self.tasks, self.args.scale1)

    def on_epoch_end(self, epoch):
        if self.args.mo_method == 'AMORE_EPO':
            print(f'EPO: descent moves={self.epo.descent}, linear-scalarization fallbacks={self.epo.n_linscalar_adjusts}')
        if not self.track_surrogates:
            return
        for name, values in self.epoch_values.items():
            if not values['approx']:
                continue
            approx, true = np.array(values['approx']), np.array(values['true'])
            h = self.surrogate_history[name]
            h['approx_mean'].append(float(approx.mean()))
            h['true_mean'].append(float(true.mean()))
            h['gap_mean'].append(float((approx - true).mean()))
            h['abs_gap_mean'].append(float(np.abs(approx - true).mean()))
            h['pearson'].append(pearson_corr(approx, true))
            h['spearman'].append(spearman_corr(approx, true))
            print(f"[Epoch {epoch + 1}] {name} approx={h['approx_mean'][-1]:.6f} true={h['true_mean'][-1]:.6f} "
                  f"gap={h['gap_mean'][-1]:.6f} pearson={h['pearson'][-1]:.4f} spearman={h['spearman'][-1]:.4f}")


class MultiFR(Backbone):
    """MultiFR: user-side ('u') and item-side ('i') fairness, weighted with MGDA."""
    needs_sampled_items = True
    MODES = 'rui'

    @classmethod
    def validate(cls, args):
        super().validate(args)
        if args.ranker != 'base':
            raise NotImplementedError(f"MultiFR supports only ranker 'base' (got '{args.ranker}')")

    def __init__(self, args, data, cli):
        super().__init__(args, data, cli)
        self.tasks = []
        if 'r' in args.mode:
            self.tasks.append('1')
        if 'u' in args.mode:
            self.tasks.append('2')
        if 'i' in args.mode:
            self.tasks.append('3')
        self.gender_label = torch.zeros(len(data.index_F) + len(data.index_M), device=args.device)
        self.gender_label[data.index_F] = 1.0
        self.target_exposure = (torch.ones(data.genre_num, 1) * float(1 / data.genre_num)).to(args.device)
        self.all_users = torch.arange(data.user_size, dtype=torch.long, device=args.device)

    def setup(self, model):
        super().setup(model)
        self.ranker = SmoothRank(temp=self.args.temp)
        self.dcg_loss = SmoothDCGLoss(args=self.args, topk=50, temp=self.args.temp)

    def compute_losses(self, model, batch):
        loss = {'1': self.rec_loss(model, batch)}
        u = batch.unique_u
        if 'u' in self.args.mode or 'i' in self.args.mode:
            scores_all = model.predict(self.all_users)
            scores = torch.gather(scores_all, 1, self.sampled_ids)

        if 'u' in self.args.mode:
            ndcg = self.dcg_loss(scores[u], scores_all[u], self.labels[u])
            mask_F = self.gender_label[u].float()
            mask_M = 1.0 - mask_F
            ndcg_F = ndcg[torch.where(mask_F == 1)[0]].sum(dim=0) / mask_F.sum()
            ndcg_M = ndcg[torch.where(mask_M == 1)[0]].sum(dim=0) / mask_M.sum()
            loss['2'] = torch.abs(torch.log(1 + torch.abs(ndcg_F - ndcg_M))).sum()
        else:
            loss['2'] = self.zero()

        if 'i' in self.args.mode:
            ranks = self.ranker(scores[u], scores_all[u])
            exposure = torch.pow(self.args.gamma, ranks)
            prob = F.gumbel_softmax(scores[u], tau=1, hard=False)
            sys_exposure = exposure * prob
            genre_top_mask = self.data.genre_mask[:, self.sampled_ids[u].long()]
            genre_exposure = torch.matmul(genre_top_mask.reshape(self.data.genre_num, -1),
                                          sys_exposure.reshape(-1, 1))
            genre_exposure = genre_exposure / genre_exposure.sum()
            loss['3'] = torch.abs(torch.log(1 + torch.abs(genre_exposure - self.target_exposure))).sum()
        else:
            loss['3'] = self.zero()
        return loss

    def weights(self, model, loss):
        return mgda_weights(model, self.optimizer, loss, self.tasks, self.args.type, self.args.device)


class Ada2Fair(Backbone):
    """Ada2Fair: stage I learns user-item fairness weights (provider 'p', user 'u'),
    stage II trains the backbone with the weighted recommendation loss."""
    MODES = 'rup'

    def __init__(self, args, data, cli):
        super().__init__(args, data, cli)
        self.tasks = []
        if 'r' in args.mode:
            self.tasks.append('1')
        if 'u' in args.mode:
            self.tasks.append('2')
        if 'p' in args.mode:
            self.tasks.append('3')

    def wrap(self, backbone):
        return Ada2FairModel(
            backbone=backbone,
            train_matrix=self.data.train_matrix,
            train_user_list=self.data.train_user_list,
            provider_ids=self.data.dataset['provider_ids'],
            args=self.args
        ).to(self.args.device)

    def setup(self, model):
        self.optimizer = torch.optim.Adam(model.backbone_parameters(), lr=self.args.lr)
        self.weight_optimizer = torch.optim.Adam(model.weight_parameters(), lr=self.args.weight_lr)

    def on_epoch_start(self, model, sampler, num_batches):
        # Stage I: learn the adaptive fairness weights
        model.update_fairness_targets()
        for _ in range(self.args.weight_epochs):
            for _ in range(num_batches):
                user, _, _ = sampler.next_batch()
                user_id = torch.from_numpy(user).long().to(self.args.device)
                loss_p, loss_u = model.weight_loss(user_id)
                weight_loss = 0.0
                if 'p' in self.args.mode:
                    weight_loss = weight_loss + loss_p
                if 'u' in self.args.mode:
                    weight_loss = weight_loss + loss_u
                self.weight_optimizer.zero_grad()
                weight_loss.backward()
                self.weight_optimizer.step()
        model.export_fairness_weight_matrix()

    def rec_loss(self, model, batch):
        if 'r' in self.args.mode:
            return model.weighted_recommendation_loss(batch.user_id, batch.pos_id, batch.neg_id)
        return self.zero()

    def compute_losses(self, model, batch):
        loss = {'1': self.rec_loss(model, batch)}
        # Stage I losses are only monitored here, they are not optimized in stage II
        with torch.no_grad():
            loss_p, loss_u = model.weight_loss(batch.user_id)
        loss['2'] = loss_u if 'u' in self.args.mode else self.zero()
        loss['3'] = loss_p if 'p' in self.args.mode else self.zero()
        return loss

    def weights(self, model, loss):
        return {t: 1.0 for t in self.tasks}

    def step(self, model, loss, scale, history):
        for t in self.tasks:
            history[f'loss_{t}'].append((loss[t].item(), loss[t].item()))
        batch_loss = loss['1']
        history['batch_loss'].append(batch_loss.item())
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

    def log_weights(self, model, scale):
        print('Ada2Fair weights summary:')
        if model.provider_fairness_weight is not None:
            p = model.provider_fairness_weight.detach().cpu().numpy()
            print(f'Provider -> min: {p.min():.4f}, max: {p.max():.4f}, mean: {p.mean():.4f}')
            providers = model.provider_ids.detach().cpu().numpy()
            print(f'Head avg: {p[providers == 0].mean():.4f}')
            print(f'Tail avg: {p[providers == 1].mean():.4f}')
        if model.user_fairness_weight is not None:
            u = model.user_fairness_weight.detach().cpu().numpy()
            print(f'User     -> min: {u.min():.4f}, max: {u.max():.4f}, mean: {u.mean():.4f}')
        if model.fairness_weight_matrix is not None:
            w = model.fairness_weight_matrix.detach().cpu().numpy()
            print(f'Final w  -> min: {w.min():.4f}, max: {w.max():.4f}, mean: {w.mean():.4f}')


METHODS = {
    'None': Backbone,
    'AMORE_SCALE': Amore, 'AMORE_MGDA': Amore, 'AMORE_EPO': Amore,
    'AMORE_ABL': Amore, 'AMORE_ABL_WOS': Amore, 'AMORE_ABL_WOZ': Amore,
    'multifr': MultiFR,
    'ADA2FAIR': Ada2Fair,
}
BACKBONES = ['BPRMF', 'LightGCN', 'NGCF', 'MixRec']


# ======================================================================
# Data
# ======================================================================

def load_data(settings, device):
    dataset, index_F, index_M, genre_mask, popular_dict, vec_pop, long_tail, short_head, train_aplt, \
        train_user_tail_list = preprocessing(settings)

    provider_ids = np.zeros(dataset['train_matrix'].shape[1], dtype=np.int64)
    provider_ids[np.array(long_tail, dtype=np.int64)] = 1
    dataset['provider_ids'] = provider_ids

    train_user_list, val_user_list = dataset['train_user_list'], dataset['val_user_list']
    max_length = max(len(i) + len(j) for i, j in zip(train_user_list, val_user_list))
    print("max_train_val_length:", max_length)
    if settings['data'] in ['ml-1m', 'ml-100k', 'facebook_books', 'amazon_baby', 'amazon_boys_girls',
                            'amazon_music']:
        max_pos = min(max_length, 200)
    else:
        max_pos = min(max_length, 100)
    print("max_pos:", max_pos)

    genre_mask = genre_mask.to(device)
    return SimpleNamespace(
        dataset=dataset,
        user_size=dataset['user_size'],
        item_size=dataset['item_size'],
        train_matrix=dataset['train_matrix'],
        train_user_list=train_user_list,
        val_user_list=val_user_list,
        train_user_tail_list=train_user_tail_list,
        index_F=index_F,
        index_M=index_M,
        genre_mask=genre_mask,
        genre_num=genre_mask.shape[0],
        long_tail=long_tail,
        max_pos=max_pos,
    )


def sample_items_per_user(data, device):
    """For each user: up to max_pos training positives, padded with negatives (labels 1/0).

    The long-tail lists are not used anymore, but they are still sampled in the same loop
    so that the random stream (and thus sampled_ids) matches the previous mains.
    """
    max_pos, item_size = data.max_pos, data.item_size
    sampled_ids = np.zeros((data.user_size, max_pos), dtype=np.int64)
    labels = np.zeros((data.user_size, max_pos), dtype=np.float32)
    for i in range(data.user_size):
        train_items = data.train_user_list[i]
        if len(train_items) > max_pos:
            pos = [train_items[j] for j in np.random.choice(len(train_items), size=max_pos, replace=False)]
        else:
            pos = train_items
        neg = negsamp_vectorized_bsearch_preverif(np.array(train_items), item_size, n_samp=max_pos - len(pos))
        tail_items = data.train_user_tail_list[i]
        if len(tail_items) > max_pos:
            tail_pos = [tail_items[j] for j in np.random.choice(len(tail_items), size=max_pos, replace=False)]
        else:
            tail_pos = tail_items
        negsamp_vectorized_bsearch_preverif(np.array(train_items), item_size, n_samp=max_pos - len(tail_pos))

        sampled_ids[i][:len(pos)] = np.array(pos)
        sampled_ids[i][len(pos):] = neg
        labels[i][:len(pos)] = 1
    return torch.LongTensor(sampled_ids).to(device), torch.LongTensor(labels).to(device)


# ======================================================================
# Training
# ======================================================================

def make_batch(user, pos, neg, device):
    return SimpleNamespace(
        unique_u=torch.as_tensor(np.unique(user), dtype=torch.long, device=device),
        user_id=torch.from_numpy(user).long().to(device),
        pos_id=torch.from_numpy(pos).long().to(device),
        neg_id=torch.from_numpy(np.squeeze(neg)).long().to(device),
    )


def train(args, exp_id, val_best, data, experiment, cli):
    t1 = time.time()
    user_neg_items = neg_item_pre_sampling(data.train_matrix, num_neg_candidates=500)
    print("Pre sampling time:{}".format(time.time() - t1))

    method = METHODS[args.mo_method](args, data, cli)
    model = method.wrap(build_backbone(args, data))
    method.setup(model)

    sampler = NegSampler(data.train_matrix, {'user_neg_items': user_neg_items},
                         batch_size=args.batch_size, num_neg=1, n_workers=4)
    num_batches = data.train_matrix.count_nonzero() // args.batch_size

    if method.needs_sampled_items:
        method.set_sampled_items(*sample_items_per_user(data, args.device))

    history_losses = {'batch_loss': []}
    for t in method.tasks:
        history_losses[f'loss_{t}'] = []
    validation_scores = []

    patience = experiment.get('patience')
    early_stopper = EarlyStopping(patience=patience, verbose=True) if patience else None
    local_val_best = -float('inf')

    try:
        epoch_times = []
        for iter in range(args.n_epochs):
            print("Epoch:", iter + 1)
            start_epoch = time.time()
            model.train()
            method.on_epoch_start(model, sampler, num_batches)

            for _ in tqdm(range(num_batches), desc='Batch Progress Bar'):
                batch = make_batch(*sampler.next_batch(), args.device)
                loss = method.compute_losses(model, batch)
                scale = method.weights(model, loss)
                method.step(model, loss, scale, history_losses)

            method.on_epoch_end(iter)
            epoch_times.append(time.time() - start_epoch)
            print('Epoch time: {:.6f}'.format(epoch_times[-1]))
            print('***** Weights Values *****')
            method.log_weights(model, scale)
            print('***** Loss Values *****')
            print('\n'.join('\'{:s}\': {:.10f}'.format(k, float(loss[k])) for k in method.tasks))

            if (iter + 1) % args.every == 0:
                print('***** Generate list of recommendation *****')
                pred_list = generate_pred_list(model, data.train_matrix, args.device, topk=50)
                print('***** Saving list of recommendation *****')
                rec_to_elliot(iter + 1, pred_list, data.dataset, exp_id, args.data)
                print('***** Accuracy performance on Validation Set *****')
                val_metric = compute_metrics(data.val_user_list, pred_list, args.metric, data.item_size)
                if args.mo_method == 'None' and val_metric > val_best:
                    val_best = val_metric
                    save_backbone_arrays(model, args, exp_id, data)
                print(f'Validation metric: {args.metric}, Value: {val_metric}')
                validation_scores.append((iter + 1, val_metric))
                local_val_best = max(local_val_best, val_metric)

                if early_stopper is not None:
                    early_stopper.check_early_stop(val_metric)
                    if early_stopper.stop_training:
                        print(f"Early stopping triggered at epoch {iter + 1}")
                        print(f"Best local {args.metric}: {local_val_best:.6f}")
                        break

        print(epoch_times)
        print(f'TRAINING TIME: {sum(epoch_times)}')
        print(f'MEAN TRAINING TIME: {sum(epoch_times) / len(epoch_times)}')
        os.makedirs(f'results/{args.data}/losses', exist_ok=True)
        with open(f'results/{args.data}/losses/{exp_id}_loss.pkl', 'wb') as f:
            pickle.dump(history_losses, f)
        surrogate_history = getattr(method, 'surrogate_history', None)
        if surrogate_history is not None:
            os.makedirs(f'results/{args.data}/surrogate_history', exist_ok=True)
            with open(f'results/{args.data}/surrogate_history/{exp_id}_surrogate.pkl', 'wb') as f:
                pickle.dump(surrogate_history, f)
        return validation_scores, val_best
    except KeyboardInterrupt:
        sys.exit()
    finally:
        sampler.close()


def build_args(settings, experiment, config_path):
    misplaced = [k for k in experiment if k.startswith('atk_')]
    if misplaced:
        raise ValueError(f"{config_path}: {misplaced} must be defined inside 'atk' "
                         f"(e.g. atk: [{{atk_con: 20, atk_div: 20}}])")
    # AMORe extensions (entropy/diversity/novelty) use the Namespace_nd conventions of the
    # amore_with_*.py scripts, so that the experiment identifiers stay the same.
    extension = str(settings['wrapper']).startswith('AMORE') and any(m in str(experiment.get('mode', ''))
                                                                     for m in 'dne')
    try:
        args = (NamespaceND if extension else Namespace)(settings, experiment)
    except (KeyError, TypeError) as e:
        raise ValueError(f"{config_path}: hyperparameters not valid for backbone '{settings['baseline']}' "
                         f"and wrapper '{settings['wrapper']}' ({type(e).__name__}: {e})") from e
    if extension:
        # Namespace_nd reads atk_con/atk_div/atk_nov/atk_ent (not for AMORE_MGDA): fill in what is
        # missing, and also accept the atk_cons/atk_prov spelling of the standard configs.
        atk = experiment.get('atk') if isinstance(experiment.get('atk'), dict) else {}
        for attr, keys in [('atk_con', ['atk_con', 'atk_cons']), ('atk_pro', ['atk_prov']),
                           ('atk_div', ['atk_div']), ('atk_nov', ['atk_nov']), ('atk_ent', ['atk_ent'])]:
            found = [atk[k] for k in keys if k in atk]
            if not hasattr(args, attr) and found:
                setattr(args, attr, found[0])
        for key in ['item_feature_path', 'entropy_mask_train']:
            if not hasattr(args, key) and key in experiment:
                setattr(args, key, experiment[key])
    METHODS[args.mo_method].validate(args)
    return args


if __name__ == '__main__':
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    cli = parse_args()
    config_path = 'config_files/' + cli.config
    with open(config_path, 'r') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    settings = conf['setting']
    if settings['baseline'] not in BACKBONES:
        raise ValueError(f"Backbone not supported: {settings['baseline']} (supported: {BACKBONES})")
    if str(settings['wrapper']) not in METHODS:
        raise ValueError(f"Wrapper not supported: {settings['wrapper']} (supported: {list(METHODS)})")
    settings['wrapper'] = str(settings['wrapper'])

    keys, values = zip(*conf['hyperparameters'].items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Fail fast on invalid hyperparameters, before loading the data
    for experiment in experiments:
        build_args(settings, experiment, config_path)

    device = torch.device('cuda:' + str(settings['gpu_id']) if torch.cuda.is_available() else 'cpu')
    data = load_data(settings, device)
    print("device:", settings['gpu_id'])
    print("Data name:", settings['data'])
    print("Total number of experiments: ", len(experiments))

    val_best = 0
    end = cli.end if cli.end is not None else len(experiments) + 1
    order = list(enumerate(experiments, start=1))
    if cli.reverse:
        order = order[::-1]
    head_id = None
    for i, experiment in order:
        if not cli.start <= i <= end:
            continue
        print(f"Experiment {i}/{len(experiments)}")
        args = build_args(settings, experiment, config_path)
        head_id = exp_setting(args)
        perf_dir = f'results/{args.data}/performance'
        os.makedirs(perf_dir, exist_ok=True)
        store_path = f'{perf_dir}/{head_id}_validation.pkl'
        store_validation = {}
        if os.path.exists(store_path):
            with open(store_path, 'rb') as f:
                store_validation = pickle.load(f)
        exp_id = exp_string(i, args)
        print("Training identifier:", exp_id)
        os.makedirs(f'results/{args.data}/parameters', exist_ok=True)
        with open(f"results/{args.data}/parameters/{exp_id}_params.txt", "a") as f:
            for arg, value in sorted(vars(args).items()):
                f.write(f"{str(arg)}\t{str(value)}\n")
        print("**** PARAMETERS SAVED ****")

        val_scores, val_best = train(args, exp_id, val_best, data, experiment, cli)
        store_validation[exp_id] = val_scores
        with open(store_path, 'wb') as f:
            pickle.dump(store_validation, f)
        print(val_scores)

    if head_id is None:
        print('No experiment in the selected range.')
        sys.exit()
    with open(f'results/{settings["data"]}/performance/{head_id}_validation.pkl', 'rb') as f:
        store_validation = pickle.load(f)
    best = {k: max(v, key=lambda x: x[1]) for k, v in store_validation.items() if v}
    maxKey = max(best, key=lambda k: best[k][1])
    print(f'maxKey: {maxKey}')
    print(f'maximumValue: {best[maxKey]}')
