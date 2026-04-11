import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import logging
import sys
import time
from argparse import ArgumentParser
from model.mf import MatrixFactorization
from model.lightgcn import LightGCNModel
from model.ngcf import NGCFModel
from SoftRank import SmoothDCGLoss, SmoothRank
from sampler import NegSampler, negsamp_vectorized_bsearch_preverif
from min_norm_solvers import MinNormSolver, gradient_normalizers
from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k, idcg_k
from preprocess import generate_rating_matrix, preprocessing
import itertools
from collections import Counter, OrderedDict
import pandas as pd
import yaml
import os
import warnings
import random
from tqdm import tqdm
from Namespace import Namespace
import math
import pickle
from imle.aimle import aimle
from imle.target import AdaptiveTargetDistribution, TargetDistribution
from torch.nn import Sigmoid
from epo_lp import EPO_LP
from early_stopping import EarlyStopping


def rank(seq: Tensor) -> Tensor:
    res = torch.argsort(torch.argsort(seq, dim=1, descending=True)) + 1
    return res.float()


# Adaptive Implicit MLE (https://arxiv.org/abs/2209.04862, AAAI 2023)
target_distribution = AdaptiveTargetDistribution(beta_update_step=1e-2)


# Implicit MLE (https://arxiv.org/abs/2106.01798, NeurIPS 2021)
# target_distribution = TargetDistribution(alpha=1.0, beta=100.0)


@aimle(target_distribution=target_distribution)
def differentiable_ranker(weights_batch: Tensor) -> Tensor:
    return rank(weights_batch)


class AIMLE_ranking:
    def __init__(self):
        pass

    def __call__(self,
                 input: Tensor) -> Tensor:
        ranks_2d = differentiable_ranker(input)
        return ranks_2d


warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable


def parse_args():
    parser = ArgumentParser(description="A frameowrk for MORS")
    parser.add_argument('--config', type=str)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=None)

    return parser.parse_args()


def generate_pred_list(model, train_matrix, topk=20):
    num_users = train_matrix.shape[0]
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    pred_list = None

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(device)

        rating_pred = model.predict(batch_user_ids)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0
        batch_raw_score = rating_pred

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        batch_score_list = rating_pred[np.arange(len(rating_pred))[:, None], batch_pred_list]

        if batchID == 0:
            pred_list = batch_pred_list
            score_list = batch_score_list
            raw_score_list = batch_raw_score
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            score_list = np.append(score_list, batch_score_list, axis=0)
            raw_score_list = np.append(raw_score_list, batch_raw_score, axis=0)

    return pred_list, score_list, raw_score_list


def compute_metrics(test_set, pred_list, metric):
    metric, k = metric.split('@')[0], int(metric.split('@')[1])
    if metric == 'ndcg':
        return ndcg_k(test_set, pred_list, k)
    elif metric == 'recall':
        return recall_at_k(test_set, pred_list, k)
    elif metric == 'precision':
        return precision_at_k(test_set, pred_list, k)
    elif metric == 'map':
        return mapk(test_set, pred_list, k)
    '''
    precision, recall, MAP, ndcg = [], [], [], []
    # for k in [5, 10, 15, 20]:
    precision.append(precision_at_k(test_set, pred_list, k))
    recall.append(recall_at_k(test_set, pred_list, k))
    MAP.append(mapk(test_set, pred_list, k))
    ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg
    '''


def neg_item_pre_sampling(train_matrix, num_neg_candidates=500):
    num_users, num_items = train_matrix.shape
    user_neg_items = []
    for user_id in range(num_users):
        pos_items = train_matrix[user_id].indices
        u_neg_item = negsamp_vectorized_bsearch_preverif(pos_items, num_items, num_neg_candidates)
        user_neg_items.append(u_neg_item)
    user_neg_items = np.asarray(user_neg_items)

    return user_neg_items


def statistics_occurrence(top_id, popular_dict):
    pop_occurrence = []
    ind_occurrence = []
    for k in [5, 10, 15, 20]:
        merged_id = list(itertools.chain(*top_id[:, :k]))
        pop_flat = list((pd.Series(merged_id)).map(popular_dict))
        count_genre = sorted(Counter(pop_flat).most_common(), key=lambda tup: tup[0])
        pop_occurrence.append([x[1] for x in count_genre])

        count_ind = Counter(merged_id).most_common()
        ind_occurrence.append([x[1] for x in count_ind])
    return pop_occurrence, ind_occurrence


def conv_mapping(mapping, x):
    for k, v in mapping.items():
        if v == x:
            return k

def rec_to_elliot(iter, top200_id, dataset, exp_string, data_name):

    num_users, top_k = top200_id.shape

    users = np.repeat(np.arange(num_users), top_k)
    items = top200_id.flatten()
    scores = np.tile(np.arange(top_k, 0, -1), num_users)

    df = pd.DataFrame({
        'user': users,
        'item': items,
        'rating': scores
    })

    user_map = pd.Series(dataset['user_mapping_inv'])
    item_map = pd.Series(dataset['item_mapping_inv'])

    df['user'] = df['user'].map(user_map)
    df['item'] = df['item'].map(item_map)
    if df['user'].isnull().any() or df['item'].isnull().any():
        raise ValueError("Failed Mapping. NaN")

    output_dir = f'results/{data_name}/recs'
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(f'{output_dir}/{exp_string}_it={iter}_recs.tsv',
              sep='\t', index=False, header=False)

def rec_to_elliot_old(iter, top200_id, dataset, exp_string):
    rec_elliot = []
    for i in range(top200_id.shape[0]):
        for j in range(top200_id.shape[1]):
            rec_elliot.append([i, top200_id[i, j], top200_id.shape[1] - j])
    rec_elliot = pd.DataFrame(rec_elliot, columns=['user', 'item', 'rating'])
    rec_elliot['user'] = rec_elliot['user'].map(lambda x: conv_mapping(dataset['user_mapping'], x))
    rec_elliot['item'] = rec_elliot['item'].map(lambda x: conv_mapping(dataset['item_mapping'], x))
    if not os.path.exists(f'results/{args.data}/recs'):
        os.makedirs(f'results/{args.data}/recs')
    rec_elliot.to_csv(f'results/{args.data}/recs/{exp_string}_it={iter}_recs.tsv',
                      sep='\t', index=False, header=False)




def exp_string(i, args):
    head = '-'.join(f'{key}={value}' for key, value in vars(args).items() if key in ['backbone', 'mo_method', 'mode'])
    tail = '-'.join(f'{key}={value}' for key, value in vars(args).items()
                    if key not in ['backbone', 'mo_method', 'mode', 'device', 'every', 'metric']).replace('.', '$')
    tail_reduced = '-'.join(f'{key}={value}' for key, value in vars(args).items()
                            if key not in ['backbone', 'mo_method', 'mode', 'device', 'every', 'metric', 'batch_size',
                                           'n_epochs', 'ranker', 'atk']).replace('.', '$')
    return str(i) + '-' + head + '-' + tail_reduced


def exp_setting(i, setting):
    return '-'.join(f'{key}={value}' for key, value in vars(setting).items()
                    if key in ['backbone', 'mo_method', 'mode', 'data'])


def normalize_loss(data):
    sigmoid = Sigmoid()
    mean = torch.mean(data)
    # print(mean)
    std_dev = torch.std(data)
    # print(std_dev)
    z_scores = (data - mean) / std_dev
    z_scores = sigmoid(z_scores)
    # utopia_point = 0
    # norm_utopia_point = (utopia_point -mean)/std_dev
    # print(norm_utopia_point)
    return z_scores  # , (norm_utopia_point-z_scores)

def normalize_loss_wo_sigmoid(data):
    # sigmoid = Sigmoid()
    mean = torch.mean(data)
    # print(mean)
    std_dev = torch.std(data)
    # print(std_dev)
    z_scores = (data - mean) / std_dev
    # z_scores = sigmoid(z_scores)
    # utopia_point = 0
    # norm_utopia_point = (utopia_point -mean)/std_dev
    # print(norm_utopia_point)
    return z_scores  # , (norm_utopia_point-z_scores)

def normalize_loss_wo_zeta(data):
    sigmoid = Sigmoid()
    # mean = torch.mean(data)
    # print(mean)
    # std_dev = torch.std(data)
    # print(std_dev)
    # z_scores = (data - mean) / std_dev
    data = sigmoid(data)
    # utopia_point = 0
    # norm_utopia_point = (utopia_point -mean)/std_dev
    # print(norm_utopia_point)
    return data  # , (norm_utopia_point-z_scores)

def compute_differentiable_ndcg(args, ranks_prov, sampled_ids_batch, labels_batch):
    """
    ranks_prov: [B, num_items] dove B = len(unique_u)
    sampled_ids_batch: [B, max_pos]
    labels_batch: [B, max_pos]
    """
    idcg = sum(1.0 / math.log(i + 2, 2) for i in range(args.atk_con))

    gathered_ranks = ranks_prov.gather(1, sampled_ids_batch)
    dcg_num = ((torch.tanh(-gathered_ranks + args.atk_con) + 1) / 2) * labels_batch.float()
    dcg = torch.sum(dcg_num / torch.log2(gathered_ranks + 1), dim=-1)

    return dcg / idcg

def compute_differentiable_aplt(ranks_prov, args, long_tail):
    ranks = (torch.tanh(-ranks_prov + args.atk_pro) + 1) / 2
    return torch.sum(ranks[:, long_tail], dim=1) / args.atk_pro



def train(args, exp_id, val_best):
    # pre-sample a small set of negative samples
    evaluation = True
    t1 = time.time()
    user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
    pre_samples = {'user_neg_items': user_neg_items}

    print("Pre sampling time:{}".format(time.time() - t1))

    gender_label = np.zeros(len(index_F) + len(index_M))
    for ind in index_F:
        gender_label[ind] = 1

    if args.backbone == 'BPRMF':
        model = MatrixFactorization(user_size, item_size, args)
        # optimizer = torch.optim.Adam(model.myparameters, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.myparameters, lr=args.lr)  # , weight_decay=args.weight_decay)
    elif args.backbone == 'LightGCN':
        model = LightGCNModel(user_size, item_size, args, dataset['train_matrix'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.backbone == 'NGCF':
        model = NGCFModel(user_size, item_size, args, dataset['train_matrix'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print("Backbone not supported.")
        return -1
    # optimizer = torch.optim.Adam(model.myparameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.mo_method != 'None':
        rank_mode = args.ranker  # or base
    else:
        rank_mode = 'None'
    if rank_mode == 'base':
        ranker = SmoothRank(temp=args.temp)
        dcg_loss = SmoothDCGLoss(args=args, topk=50, temp=args.temp)
    elif rank_mode == 'AIMLE':
        ranker = AIMLE_ranking()

    sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=1, n_workers=4)

    # computing number of batches
    num_batches = train_matrix.count_nonzero() // args.batch_size

    loss = {}
    scale = {}
    loss_clone = {}
    loss_data = {}
    grads = {}
    tasks = []

    if args.mo_method in ['AMORE_MGDA', 'AMORE_EPO', 'AMORE_SCALE', 'AMORE_ABL', 'AMORE_ABL_WOS', 'AMORE_ABL_WOZ']:
        if 'r' in args.mode:
            tasks.append('1')
        if 's' in args.mode:
            tasks.append('4')
        else:
            if 'm' in args.mode:
                tasks.append('2')
            if 'p' in args.mode:
                tasks.append('3')

    elif args.mo_method == 'multifr':
        if 'r' in args.mode:
            tasks.append('1')
        if 'u' in args.mode:
            tasks.append('2')
        if 'i' in args.mode:
            tasks.append('3')
    else:
        tasks.append('1')

    if args.mo_method in ['AMORE_EPO']:
        _, n_params = getNumParams(model.parameters())
        # preference = np.abs(np.random.randn(1, len(tasks)))
        preference = np.array([args.scale1, 1 - args.scale1, 1 - args.scale1])
        # preference /= preference.sum(axis=1, keepdims=True)
        epo_lp = EPO_LP(m=len(tasks), n=n_params, r=preference)

    history_losses = {'batch_loss': []}
    for t in tasks:
        history_losses[f'loss_{t}'] = []

    validation_scores = []

    # Target exposure on movie genres:
    target_exposure = (torch.ones(genre_num, 1) * float(1 / genre_num)).to(args.device)

    # Sample max_pos items for each user
    neg_ids_list = []
    pos_ids_list = []
    neg_ids_for_tail_list = []
    pos_ids_for_tail_list = []
    # sampling positive items (if needed) based on train set
    for i in range(user_size):
        if (len(train_user_list[i]) > max_pos):
            sampled_pos_ids = np.random.choice(len(train_user_list[i]), size=max_pos, replace=False)
            tmp = [train_user_list[i][j] for j in sampled_pos_ids]
            pos_ids_list.append(tmp)
        else:
            pos_ids_list.append(train_user_list[i])
        neg_ids_list.append(negsamp_vectorized_bsearch_preverif(np.array(train_user_list[i]), item_size,
                                                                n_samp=max_pos - len(pos_ids_list[i])))
        if (len(train_user_tail_list[i]) > max_pos):
            sampled_pos_tail_ids = np.random.choice(len(train_user_tail_list[i]), size=max_pos, replace=False)
            tmp = [train_user_tail_list[i][j] for j in sampled_pos_tail_ids]
            pos_ids_for_tail_list.append(tmp)
        else:
            pos_ids_for_tail_list.append(train_user_tail_list[i])
        neg_ids_for_tail_list.append(negsamp_vectorized_bsearch_preverif(np.array(train_user_list[i]), item_size,
                                                                         n_samp=max_pos - len(
                                                                             pos_ids_for_tail_list[i])))

    sampled_ids = np.zeros((user_size, max_pos), dtype=np.int64)
    sampled_tail_ids = np.zeros((user_size, max_pos), dtype=np.int64)
    labels = np.zeros((user_size, max_pos), dtype=np.float32)
    labels_tail = np.zeros((user_size, max_pos), dtype=np.float32)

    for i in range(user_size):
        sampled_ids[i][:len(pos_ids_list[i])] = np.array(pos_ids_list[i])
        sampled_tail_ids[i][:len(pos_ids_for_tail_list[i])] = np.array(pos_ids_for_tail_list[i])
        sampled_ids[i][len(pos_ids_list[i]):] = neg_ids_list[i]
        sampled_tail_ids[i][len(pos_ids_for_tail_list[i]):] = neg_ids_for_tail_list[i]

        labels[i][:len(pos_ids_list[i])] = 1
        labels_tail[i][:len(pos_ids_for_tail_list[i])] = 1

    sampled_ids = torch.LongTensor(sampled_ids).to(args.device)
    labels = torch.LongTensor(labels).to(args.device)
    sampled_tail_ids = torch.LongTensor(sampled_tail_ids).to(args.device)
    labels_tail = torch.LongTensor(labels_tail).to(args.device)
    # results = []
    # validation_results = []

    users = []
    for k, v in dataset['user_mapping'].items():
        users.append(v)
    try:
        epoch_times = []
        for iter in range(args.n_epochs):
            # start_time = time.time()
            # loss['3'] = torch.tensor(0)
            # acc = torch.tensor(0)
            # acc_ndcg = torch.tensor(0)
            print("Epoch:", iter + 1)
            if args.mo_method in ['AMORE_EPO']:
                n_linscalar_adjusts = 0
                descent = 0.

            start_epoch = time.time()
            model.train()

            # Start Training
            for _ in tqdm(range(num_batches), desc='Batch Progress Bar'):
                if args.mo_method in ['AMORE_EPO']:
                    grads = {}
                    losses = []
                # start_batch = time.time()
                # print("Batch: ", batch_id)
                user, pos, neg = sampler.next_batch()
                neg = np.squeeze(neg)
                unique_u = torch.LongTensor(list(set(user.tolist())))

                user_id = torch.from_numpy(user).type(torch.LongTensor).to(args.device)
                pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(args.device)
                neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(args.device)

                # Backbone Model Loss
                if 'r' in args.mode:
                    loss['1'] = model(user_id, pos_id, neg_id)
                else:
                    loss['1'] = torch.tensor(0)

                # AMORe Method
                if args.mo_method in ['AMORE_MGDA', 'AMORE_EPO', 'AMORE_SCALE', 'AMORE_ABL', 'AMORE_ABL_WOS', 'AMORE_ABL_WOZ']:
                    if args.backbone == 'BPRMF':
                        # scores_all = model.myparameters[0].mm(model.myparameters[1].t())
                        batch_users = unique_u.to(args.device)
                        user_emb = model.myparameters[0][batch_users]  # [B, d]
                        item_emb = model.myparameters[1]  # [I, d]
                        scores_all = user_emb @ item_emb.t()  # [B, I]
                    elif args.backbone == 'LightGCN':
                        # scores_all = model.predict(users)
                        batch_users = unique_u.to(args.device)
                        scores_all = model.predict(batch_users)
                    elif args.backbone == 'NGCF':
                        # scores_all = model.predict(users)
                        batch_users = unique_u.to(args.device)
                        scores_all = model.predict(batch_users)
                    else:
                        raise ValueError("Backbone not supported.")

                    ranks_prov = ranker(scores_all)

                    sampled_ids_batch = sampled_ids[unique_u]
                    labels_batch = labels[unique_u]

                    assert ranks_prov.dim() == 2
                    assert sampled_ids_batch.dim() == 2
                    assert labels_batch.dim() == 2

                    assert sampled_ids_batch.min().item() >= 0
                    assert sampled_ids_batch.max().item() < ranks_prov.size(1), (
                        f"sampled_ids_batch max={sampled_ids_batch.max().item()}, "
                        f"num_items={ranks_prov.size(1)}"
                    )

                    assert ranks_prov.size(0) == sampled_ids_batch.size(0) == labels_batch.size(0), (
                        f"Mismatch batch dims: ranks={ranks_prov.size()}, "
                        f"sampled={sampled_ids_batch.size()}, labels={labels_batch.size()}"
                    )

                    if 'm' in args.mode:
                        ndcg = compute_differentiable_ndcg(args, ranks_prov, sampled_ids_batch, labels_batch)

                        if args.mo_method in ['AMORE_MGDA', 'AMORE_EPO', 'AMORE_SCALE']:
                            loss['2'] = normalize_loss(torch.square(1 - ndcg)).sum()
                        elif args.mo_method in ['AMORE_ABL_WOS']:
                            loss['2'] = normalize_loss_wo_sigmoid(torch.square(1 - ndcg)).sum()
                        elif args.mo_method in ['AMORE_ABL_WOZ']:
                            loss['2'] = normalize_loss_wo_zeta(torch.square(1 - ndcg)).sum()
                        else:
                            loss['2'] = torch.square(1 - ndcg).sum()
                    else:
                        loss['2'] = torch.tensor(0.0, device=args.device)
                    if 'p' in args.mode:
                        if rank_mode == 'base':
                            scores_tail = scores_all[:, long_tail]  # .to('cpu')
                            if args.ablation == 'relevant':
                                scores = torch.gather(scores_all, 1, sampled_ids).to(args.device)
                                ranks_prov = ranker.forward(scores_tail[unique_u], scores[unique_u])
                            elif args.ablation == 'popular':
                                scores = scores_all[:, short_head]
                                ranks_prov = ranker.forward(scores_tail[unique_u], scores[unique_u])
                            elif args.ablation == 'all':
                                ranks_prov = ranker.forward(scores_tail[unique_u], scores_all[unique_u])
                            ranks_prov = ranks_prov[unique_u]
                            ranks_prov = (torch.tanh(-ranks_prov + args.atk_pro) + 1) / 2
                            ranks_prov = torch.sum(ranks_prov, dim=1)
                            ranks_prov = torch.clamp(ranks_prov, min=0, max=args.atk_pro) / args.atk_pro
                            loss['3'] = (torch.square(1 - ranks_prov)).sum()
                            # loss['3'] = loss['3'] + ranks_prov
                            # acc = acc + ranks_prov/len(unique_u)
                            del ranks_prov
                        else:
                            aplt = compute_differentiable_aplt(ranks_prov, args, long_tail)


                            if args.mo_method in ['ADAAMORE','AMORE_MGDA', 'AMORE_EPO', 'AMORE_SCALE']:
                                loss['3'] = normalize_loss(torch.square(1 - aplt)).sum()
                            elif args.mo_method in ['AMORE_ABL_WOS']:
                                loss['3'] = normalize_loss_wo_sigmoid(torch.square(1 - aplt)).sum()
                            elif args.mo_method in ['AMORE_ABL_WOZ']:
                                loss['3'] = normalize_loss_wo_zeta(torch.square(1 - aplt)).sum()
                            else:
                                loss['3'] = torch.square(1 - aplt).sum()
                    else:
                        loss['3'] = torch.tensor(0)

                    if 's' in args.mode:
                        # loss['4'] = torch.log(1 + loss['2'] + loss['3']) / len(unique_u)
                        loss['4'] = (loss['2'] + loss['3']) / len(unique_u)
                        loss['2'] = torch.tensor(0)
                        loss['3'] = torch.tensor(0)
                    else:
                        loss['2'] = loss['2'] / len(unique_u)
                        loss['3'] = loss['3'] / len(unique_u)

                    # if args.mo_method in ['USERADAAMORE']:


                # MultiFR Method
                elif args.mo_method == 'multifr':
                    if 'u' in args.mode or 'i' in args.mode:
                        if args.backbone == 'BPRMF':
                            scores_all = model.myparameters[0].mm(model.myparameters[1].t())
                        elif args.backbone == 'LightGCN':
                            scores_all = model.predict(users)
                        elif args.backbone == 'NGCF':
                            scores_all = model.predict(users)
                        # scores_all[:,item_size] = -np.inf
                        scores = torch.gather(scores_all, 1, sampled_ids).to(args.device)

                    if 'u' in args.mode:
                        ndcg = dcg_loss(scores[unique_u], scores_all[unique_u], labels[unique_u])
                        # ndcg = -torch.log(ndcg)
                        # print("ndcg:", -torch.log(ndcg))
                        # loss['1'] = (1 - ndcg[:,9]).sum()
                        mask_F = gender_label[unique_u]
                        mask_M = 1 - mask_F

                        mask_F = torch.from_numpy(mask_F).type(torch.FloatTensor).to(args.device)
                        mask_M = torch.from_numpy(mask_M).type(torch.FloatTensor).to(args.device)
                        pos_F = torch.tensor(np.where(mask_F.cpu() == 1)[0]).to(args.device)
                        pos_M = torch.tensor(np.where(mask_M.cpu() == 1)[0]).to(args.device)

                        ndcg_F = ndcg[pos_F]
                        ndcg_M = ndcg[pos_M]

                        # sum the matrix in column
                        ndcg_F = ndcg_F.sum(dim=0) / mask_F.sum()
                        ndcg_M = ndcg_M.sum(dim=0) / mask_M.sum()

                        loss['2'] = torch.abs(torch.log(1 + torch.abs(ndcg_F - ndcg_M))).sum()
                        # loss['2'] = torch.abs(torch.log(ndcg_F+0.5) - torch.log(ndcg_M+0.5)).sum()
                        # loss['2'] = loss['2'] * 20
                    else:
                        loss['2'] = torch.tensor(0)

                    if 'i' in args.mode:
                        # ranks = ranker(scores_all)[unique_u]
                        ranks = ranker(scores[unique_u], scores_all[unique_u])

                        # print("ranks:", ranks.shape, ranks)
                        exposure = torch.pow(args.gamma, ranks)

                        prob = F.gumbel_softmax(scores[unique_u], tau=1, hard=False)
                        sys_exposure = exposure * prob

                        genre_top_mask = genre_mask[:, sampled_ids[unique_u].long()]
                        # print("genre_mask:", genre_mask.shape)
                        # print("genre_top_mask:", genre_top_mask.shape)

                        genre_exposure = torch.matmul(genre_top_mask.reshape(genre_num, -1),
                                                      sys_exposure.reshape(-1, 1))
                        # print("genre_exposure:", genre_exposure.shape)
                        genre_exposure = genre_exposure / genre_exposure.sum()

                        loss['3'] = torch.abs(torch.log(1 + torch.abs(genre_exposure - target_exposure))).sum()
                        # loss['3'] = loss['3'] * args.batch_size
                        # loss['3'] = torch.abs(torch.log(genre_exposure+0.5) - torch.log(target_exposure+0.5)).sum()
                        # loss['3'] = loss['3'] * 20
                    else:
                        loss['3'] = torch.tensor(0)

                # Use MOOP or not
                if args.mo_method in ['AMORE_MGDA', 'multifr']:
                    # Copy the loss data. Average loss1 for calculating scale
                    for k in loss:
                        if k == '1':
                            loss_clone[k] = loss[k].clone()  # / args.batch_size
                        elif k == '2':
                            loss_clone[k] = loss[k].clone()
                        else:
                            loss_clone[k] = loss[k].clone()

                    for t in tasks:
                        # print(t)
                        optimizer.zero_grad()
                        loss_clone[t].backward(retain_graph=True)
                        loss_data[t] = loss_clone[t].item()

                        grads[t] = []
                        for param in model.parameters():
                            if param.grad is not None:
                                tmp = Variable(param.grad.data.clone(), requires_grad=False).to(args.device)
                                tmp = tmp.flatten()
                                grads[t].append(tmp)

                    gn = gradient_normalizers(grads, loss_data, args.type)

                    for t in tasks:
                        if gn[t] == 0.0:
                            gn[t] += 1
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]

                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])
                elif args.mo_method in ['AMORE_EPO']:
                    for k in loss:
                        if k == '1':
                            loss_clone[k] = loss[k].clone()  # / args.batch_size
                        elif k == '2':
                            loss_clone[k] = loss[k].clone()
                        else:
                            loss_clone[k] = loss[k].clone()
                    for i in tasks:
                        optimizer.zero_grad()
                        # task_loss = model(X, ts)
                        losses.append(loss_clone[i].data.cpu().numpy())
                        loss_clone[i].backward(retain_graph=True)
                        loss_data[i] = loss_clone[i].item()
                        # One can use scalable method proposed in the MOO-MTL paper
                        # for large scale problem; but we use the gradient
                        # of all parameters in this experiment.
                        grads[i] = []
                        for param in model.parameters():
                            if param.grad is not None:
                                tmp = Variable(param.grad.data.clone(), requires_grad=False).to(args.device)
                                tmp = tmp.flatten()
                                grads[i].append(tmp)

                    grads_list = [torch.cat(grads[t]) for t in tasks]
                    G = torch.stack(grads_list)
                    GG = G @ G.T
                    losses = np.stack(losses)
                    try:
                        # Calculate the alphas from the LP solver
                        sol = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                        if epo_lp.last_move == "dom":
                            descent += 1
                    except Exception as e:
                        # print(e)
                        # print(f'losses:{losses}')
                        # print(f'C:\n{GG.cpu().numpy()}')
                        # raise RuntimeError('manual tweak')
                        sol = None
                    if sol is None:  # A patch for the issue in cvxpy
                        sol = preference / preference.sum()
                        n_linscalar_adjusts += 1

                    if torch.cuda.is_available():
                        sol = len(tasks) * torch.from_numpy(sol).cuda()
                    else:
                        sol = len(tasks) * torch.from_numpy(sol)
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])

                else:
                    scale = {'1': args.scale1, '2': 1.0 - args.scale1, '3': 1.0 - args.scale1, '4': 1.0 - args.scale1}


                batch_loss = 0

                for t in tasks:
                    history_losses[f'loss_{t}'].append((loss[t].item(), loss[t].item() * scale[t]))
                    batch_loss += loss[t] * scale[t]

                history_losses['batch_loss'].append(batch_loss.item())


                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()


            # if args.mo_method == 'AMORE':
            #     print(f"\nAPLT loss:\t{acc / num_batches} (the lower the better, [0,1])")
            #     print(f"Approx nDCG loss:\t{acc_ndcg / num_batches} (the lower the better, [0,1])")
            end_epoch = time.time()
            epoch_times.append(end_epoch - start_epoch)
            print('Epoch time: {:.6f}'.format(end_epoch-start_epoch))
            print('***** Weights Values *****')
            try:
                print('\n'.join('\'{:s}\': {:.10f}'.format(k, scale[k]) for k in tasks))
            except NameError:
                print('Per user adaptive weights')
            # print('bpr_loss:{:.6f}, user_disparity:{:.6f}, item_disparity:{:.6f}'.format(loss['1'].item(),
            #                                                                              loss['2'].item(),
            #                                                                              loss['3'].item()))
            print('***** Loss Values *****')
            print('\n'.join('\'{:s}\': {:.10f}'.format(k, loss[k]) for k in tasks))
            # print('bpr_loss:{:.6f}, first_loss:{:.6f}, second_loss:{:.6f}'.format(loss['1'].item(), loss['2'].item(), loss['3'].item()))

            # end = time.time()
            # print("time:{:.2f}".format(end - start))

            if (iter + 1) % args.every == 0 and evaluation == True:
                model.eval()
                # Generate list of recommendation
                print('***** Generate list of recommendation *****')
                pred_list, score_matrix, raw_score_matrix = generate_pred_list(model, train_matrix, topk=50)
                # Save list of recommendation for later use
                print('***** Saving list of recommendation *****')
                rec_to_elliot(iter + 1, pred_list, dataset, exp_id , args.data)
                # Keep track of performance on Validation Set to establish best epoch
                print('***** Accuracy performance on Validation Set *****')
                val_metric = compute_metrics(val_user_list, pred_list, args.metric)
                if args.mo_method == 'None':
                    if val_metric > val_best:
                        val_best = val_metric
                        if not os.path.exists(f'arrays/'):
                            os.makedirs(f'arrays/')
                        if not os.path.exists(f'arrays/{args.data}/'):
                            os.makedirs(f'arrays/{args.data}/')
                        # ===== Embedding =====
                        if args.backbone == 'BPRMF':
                            user_emb = model.myparameters[0].detach().cpu().numpy()
                            item_emb = model.myparameters[1].detach().cpu().numpy()

                        elif args.backbone == 'LightGCN':
                            gu, gi = model.propagate_embeddings(evaluate=True)
                            user_emb = gu.detach().cpu().numpy()
                            item_emb = gi.detach().cpu().numpy()

                        elif args.backbone == 'NGCF':
                            gu, gi = model.propagate_embeddings(evaluate=True)
                            user_emb = gu.detach().cpu().numpy()
                            item_emb = gi.detach().cpu().numpy()

                        else:
                            raise ValueError(f"Backbone not supported: {args.backbone}")
                        # ===== Score matrix user-item =====
                        score_matrix = model.predict(torch.tensor(users).to(device)).detach().cpu().numpy()

                        rows, cols = train_matrix.nonzero()
                        score_matrix[rows, cols] = -1e9

                        # ===== Saving =====
                        np.savez_compressed(
                            f'arrays/{args.data}/{args.backbone}_{args.mo_method}_{exp_id}.npz',
                            user_emb=user_emb,
                            item_emb=item_emb,
                            score_matrix=score_matrix,
                            user_map=np.array(dataset['user_mapping_inv'], dtype=object),
                            item_map=np.array(dataset['item_mapping_inv'], dtype=object),
                            metric_name=np.array([args.metric], dtype=object)
                        )

                        #np.savez_compressed(f'arrays/{args.data}/{args.backbone}_{args.mo_method}_{args.data}.npz',
                        #                    model.predict(torch.tensor(users).to(device)).cpu().detach().numpy(),
                        #                    fmt='%f')
                # precision, recall, MAP, ndcg = compute_metrics(val_user_list, pred_list, topk=20)
                print(f'Validation metric: {args.metric}, Value: {val_metric}')
                validation_scores.append((iter + 1, val_metric))
                # print('VAL Precision:', precision)
                # print('VAL Recall:', recall)
                # print('VAL MAP:', MAP)
                # print('VAL NDCG:', ndcg)
                # for k in range(0, len(precision)):
                #     val_temp = [iter, (k+1)*5, precision[k], recall[k], ndcg[k]]
                #     validation_results.append(val_temp)

                '''
                # Current performance on Test Set
                print('***** Performance on Test Set *****')
                precision, recall, MAP, ndcg = compute_metrics(test_user_list, pred_list, topk=20)
                print('TEST Precision:', precision)
                print('TEST Recall:', recall)
                print('TEST MAP:', MAP)
                print('TEST NDCG:', ndcg)

                top200_id = pred_list

                pop_occurrence, ind_occurrence = statistics_occurrence(top200_id, popular_dict)

                """
                Fairness measurement
                """
                print("Fairness on user:")
                f_u = []
                for k in [5, 10, 15, 20]:
                    f_u.append(Fairness_user(test_user_list, pred_list, index_F, index_M, user_size, item_size, topk=k))
                print("Fairness on item:")
                f_i = []
                for k in [5, 10, 15, 20]:
                    f_i.append(Fairness_item(model, genre_mask, target_exposure, pred_list, user_size, topk=k, args=args))
                print("Gini index:")
                gini = Gini(ind_occurrence)
                print("Popularity_rate:")
                pop_rate = Popularity_rate(pop_occurrence)
                print("Simpson_Diversity:")
                sim_d = Simpson_Diversity(pop_occurrence)

                for k in range(0, len(precision)):
                    part_res = [iter, (k+1)*5, precision[k], recall[k], ndcg[k], gini[k], pop_rate[k], sim_d[k], f_i[k], f_u[k]]
                    for c,v in loss.items():
                        part_res.append(v.item())
                    for c,v in scale.items():
                        part_res.append(v)
                    results.append(part_res)
                '''
                # user_neg_items = neg_item_pre_sampling(train_matrix, num_neg_candidates=500)
                # pre_samples = {'user_neg_items': user_neg_items}
                # sampler = NegSampler(train_matrix, pre_samples, batch_size=args.batch_size, num_neg=1, n_workers=4)

        '''
        columns = ['iter', 'cutoff', 'precision', 'recall', 'ndcg', 'gini', 'popularity_rate', 'simpson_diversity',
                   'item_disparity', 'user_disparity']
        for c, v in loss.items():
            columns.append(f'loss_{int(c)}')
        for c, v in scale.items():
            columns.append(f'weight_{int(c)}')
        pd.DataFrame(results, columns=columns).to_csv(f'results/{args.data}/performance/{args.backbone}_{args.mo_method}_{eventid}_performance.tsv', sep='\t',
                                     index=False)
        '''
        print(epoch_times)
        print(f'TRAINING TIME: {sum(epoch_times)}')
        print(f'MEAN TRAINING TIME: {sum(epoch_times) / len(epoch_times)}')
        if not os.path.exists(f'results/{args.data}/losses'):
            os.makedirs(f'results/{args.data}/losses')
        with open(f'results/{args.data}/losses/{exp_id}_loss.pkl', 'wb') as f:
            pickle.dump(history_losses, f)
        # val_columns = ['iter', 'cutoff', 'precision', 'recall', 'ndcg']
        # validation_results = pd.DataFrame(validation_results, columns=val_columns)
        # if not os.path.exists(f'results/{args.data}/performance'):
        #     os.makedirs(f'results/{args.data}/performance')
        # validation_results.to_csv(
        #     f'results/{args.data}/performance/{exp_id}_validation.tsv', sep='\t',
        #     index=False)
        sampler.close()
        return validation_scores, val_best
    except KeyboardInterrupt:
        sampler.close()
        sys.exit()
    # if not os.path.exists(f'results/{args.data}/parameters'):
    #    os.makedirs(f'results/{args.data}/parameters')
    # f = open(f'results/{args.data}/parameters/{exp_id}_params.txt', "a")
    # f.write("***** Best epochs: *****\n")
    # for k in [5, 10, 15, 20]:
    #     temp = validation_results[validation_results.cutoff == k]
    #     for metric in ['recall', 'ndcg']:
    #         it = temp[temp[metric] == temp[metric].max()][['iter']].values[0][0]
    #         max_met = temp[temp[metric] == temp[metric].max()][[metric]].values[0][0]
    #         f.write(f"{metric}@{str(k)}\t{str(it)}\t{str(max_met)}\n")
    # f.close()

    print('***** END EXPERIMENT *****')


if __name__ == '__main__':
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    config = parse_args()
    config.config = 'config_files/' + config.config
    with open(config.config, 'r') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    settings = conf['setting']
    keys, values = zip(*conf['hyperparameters'].items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    device = torch.device('cuda:' + str(settings['gpu_id']) if torch.cuda.is_available() else 'cpu')

    '''
    Processing of data information
    '''
    dataset, index_F, index_M, genre_mask, popular_dict, vec_pop, long_tail, short_head, train_aplt, train_user_tail_list = preprocessing(
        settings)
    # train_aplt_df = []
    # for i, u in enumerate(train_aplt):
    #     train_aplt_df.append([dataset['user_mapping_inv'][i], u])
    # train_aplt_df = pd.DataFrame(train_aplt_df)
    # train_aplt_df.to_csv(f'results/{settings["data"]}/train_aplt_df.tsv', sep='\t', index=False, header=False)

    genre_mask = genre_mask.to(device)
    popular_tuple = OrderedDict(sorted(popular_dict.items()))
    popular_list = [x[1] for x in popular_tuple.items()]
    # print("popular_tuple:", popular_tuple)
    # print("popular_list:", popular_list)

    # print("Number of females:", len(index_F))
    # print("Number of males:", len(index_M))
    # print("genre_mask:", genre_mask.shape)

    genre_num = genre_mask.shape[0]

    user_size, item_size = dataset['user_size'], dataset['item_size']
    train_user_list, val_user_list, test_user_list = dataset['train_user_list'], dataset['val_user_list'], dataset[
        'test_user_list']
    train_val_user_list = [i + j for i, j in zip(train_user_list, val_user_list)]
    # np.save(f'arrays/{settings["data"]}/long_tail.npy', np.array(long_tail, dtype=np.int64))
    # np.save(f'arrays/{settings["data"]}/train_user_list.npy', np.array(train_user_list, dtype=object))
    all_list = [i + j + k for i, j, k in zip(train_user_list, val_user_list, test_user_list)]
    train_val_list = [i + j for i, j in zip(train_user_list, val_user_list)]

    # Build the observed rating matrix
    train_matrix, val_matrix, test_matrix = dataset['train_matrix'], dataset['val_matrix'], dataset['test_matrix']
    train_val_matrix = generate_rating_matrix(train_val_user_list, user_size, item_size)

    """only consider training and testing"""
    # Other statistics
    max_all_length = 0
    for i in range(len(all_list)):
        if len(all_list[i]) > max_all_length:
            max_all_length = len(all_list[i])
    print("max_all_length:", max_all_length)

    # for training.
    max_length = 0

    for i in range(len(train_val_user_list)):
        if len(train_val_user_list[i]) > max_length:
            max_length = len(train_val_user_list[i])
    print("max_train_val_length:", max_length)
    if settings['data'] in ['ml-1m', 'ml-100k']:
        max_pos = max_length if max_length < 200 else 200
    elif settings['data'] in ['facebook_books', 'amazon_baby', 'amazon_boys_girls', 'amazon_music']:
        max_pos = max_length if max_length < 200 else 200
    else:
        max_pos = max_length if max_length < 100 else 100
    print("max_pos:", max_pos)

    print("device:", settings['gpu_id'])
    print("Data name:", settings['data'])

    # check = save_parameters()
    # if check:
    #     print("Arguments successfully saved!")
    # else:
    #     print("Arguments not saved!")

    print("Total number of experiments: ", len(experiments))
    val_best = 0
    if config.end is None:
        config.end = len(experiments) + 1
    # for i, experiment in reversed(list(enumerate(experiments, start=1))):
    for i, experiment in enumerate(experiments, start=1):
        if config.start <= i <= config.end:
            print(f"Experiment {i}/{len(experiments)}")
            # eventid = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S%f")[:-6]
            args = Namespace(settings, experiment)
            head_id = exp_setting(i, args)
            if not os.path.exists(f'results/{args.data}/performance'):
                os.makedirs(f'results/{args.data}/performance')
            if not os.path.exists(f'results/{args.data}/performance/{head_id}_validation.pkl'):
                store_validation = {}
            else:
                with open(f'results/{args.data}/performance/{head_id}_validation.pkl', 'rb') as f:
                    store_validation = pickle.load(f)
            exp_id = exp_string(i, args)
            print("Training identifier:", exp_id)
            try:
                f = open(f"results/{args.data}/parameters/{exp_id}_params.txt", "a")
                for arg, value in sorted(vars(args).items()):
                    f.write(f"{str(arg)}\t{str(value)}\n")
                f.close()
                print("**** PARAMETERS SAVED ****")
            except Exception as e:
                print(e)
                print("**** PARAMETERS NOT SAVED ****")
            val_scores, val_best = train(args, exp_id, val_best)
            store_validation[exp_id] = val_scores
            with open(f'results/{args.data}/performance/{head_id}_validation.pkl', 'wb') as f:
                pickle.dump(store_validation, f)
            print(val_scores)
    with open(f'results/{args.data}/performance/{head_id}_validation.pkl', 'rb') as f:
        store_validation = pickle.load(f)
    for k, v in store_validation.items():
        store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
    maximumValue = max(store_validation.values(), key=lambda k: k[1])
    maxKey = next(k for k, v in store_validation.items() if v == maximumValue)
    print(f'maxKey: {maxKey}')
    print(f'maximumValue: {maximumValue}')






