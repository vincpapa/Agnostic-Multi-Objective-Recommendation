import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_sparse import SparseTensor
import random


class MixRecModel(nn.Module):
    def __init__(self, num_users, num_items, args, train_matrix):
        super().__init__()
        random_seed = args.seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.num_users = num_users
        self.num_items = num_items
        self.device = args.device

        # backbone hyperparameters
        self.dim = args.dim
        self.n_layers = args.layers
        self.ssl_lambda = args.ssl_lambda
        self.alpha = args.mix_alpha
        self.temperature = args.temperature
        self.reg_weight = args.weight_decay

        # embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.dim)
        self.item_embedding = nn.Embedding(
            self.num_items + 1, self.dim, padding_idx=self.num_items
        )

        # normalized adjacency, faithful to WarpRec normalize=True
        self.adj = self.build_adj(train_matrix)

        # propagation network
        self.propagation_network = nn.ModuleList(
            [LGConv() for _ in range(self.n_layers)]
        )

        # mean pooling over propagation layers
        alpha_tensor = torch.full(
            (self.n_layers + 1,),
            1.0 / (self.n_layers + 1),
            dtype=torch.float32,
        )
        self.register_buffer("alpha_gcn", alpha_tensor)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def build_adj(self, train_matrix):
        """
        Faithful to WarpRec get_adj_mat(..., normalize=True):
        build bipartite user-item graph and apply symmetric normalization.

        Node indexing:
            users: [0, ..., num_users-1]
            items: [num_users, ..., num_users+num_items]
        where the last item node is the padding node.
        """
        num_users = self.num_users
        num_items_with_pad = self.num_items + 1
        num_nodes = num_users + num_items_with_pad

        inter = train_matrix.tocoo()

        user_idx = inter.row.astype(np.int64)
        item_idx = inter.col.astype(np.int64) + num_users

        rows = np.concatenate([user_idx, item_idx])
        cols = np.concatenate([item_idx, user_idx])
        data = np.ones(len(rows), dtype=np.float32)

        adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

        # D^{-1/2} A D^{-1/2}
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
        nonzero = deg > 0
        deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)

        d_mat_inv_sqrt = sp.diags(deg_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()

        sparse_adj = SparseTensor(
            row=torch.from_numpy(norm_adj.row).long(),
            col=torch.from_numpy(norm_adj.col).long(),
            value=torch.from_numpy(norm_adj.data).float(),
            sparse_sizes=(num_nodes, num_nodes),
        )

        return sparse_adj.to(self.device)

    def propagate_embeddings(self):
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]
        current_embeddings = ego_embeddings

        for conv_layer in self.propagation_network:
            current_embeddings = conv_layer(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        final_embeddings = torch.zeros_like(ego_embeddings)
        for k, emb in enumerate(embeddings_list):
            final_embeddings += emb * self.alpha_gcn[k]

        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items + 1], dim=0
        )
        return user_final, item_final

    def _mix_embeddings(self, original, shuffled, beta):
        return beta * original + (1.0 - beta) * shuffled

    def _collective_mixing(self, embeddings):
        batch_size = embeddings.size(0)

        dir_dist = torch.distributions.Dirichlet(
            torch.ones(batch_size, device=self.device)
        )
        coeffs = dir_dist.sample().unsqueeze(0)

        collective_view = torch.mm(coeffs, embeddings)
        return collective_view.expand(batch_size, -1)

    def _hard_nce_loss(self, anchor, positive, neg_disorder, neg_collective):
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        neg_disorder = F.normalize(neg_disorder, p=2, dim=1)
        neg_collective = F.normalize(neg_collective, p=2, dim=1)

        pos_sim = (anchor * positive).sum(dim=1) / self.temperature
        dis_sim = (anchor * neg_disorder).sum(dim=1) / self.temperature
        col_sim = (anchor * neg_collective).sum(dim=1) / self.temperature

        batch_sim_matrix = torch.mm(anchor, positive.t()) / self.temperature

        all_logits = torch.cat(
            [batch_sim_matrix, dis_sim.unsqueeze(1), col_sim.unsqueeze(1)], dim=1
        )

        loss = -pos_sim + torch.logsumexp(all_logits, dim=1)
        return loss

    def _dual_mixing_cl_loss(self, original, mixed, disordered, collective, beta):
        l_pos = self._hard_nce_loss(
            anchor=original,
            positive=mixed,
            neg_disorder=disordered,
            neg_collective=collective,
        )

        l_neg = self._hard_nce_loss(
            anchor=disordered,
            positive=mixed,
            neg_disorder=original,
            neg_collective=collective,
        )

        return (beta * l_pos + (1.0 - beta) * l_neg).mean()

    def forward(self, user, pos, neg):
        total_loss, _ = self.custom_forward(user, pos, neg)
        return total_loss

    def custom_forward(self, user, pos, neg):
        """
        Returns:
            total_loss: scalar tensor
            main_loss_vec: per-sample recommendation loss vector
        """
        batch_size = user.size(0)

        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos]
        neg_embeddings = item_all_embeddings[neg]

        # Standard BPR part
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_pos_vec = -F.logsigmoid(pos_scores - neg_scores)

        # Regularization
        reg_loss = self.reg_weight * (
            self.user_embedding(user).pow(2).sum() +
            self.item_embedding(pos).pow(2).sum() +
            self.item_embedding(neg).pow(2).sum()
        ) / batch_size

        # Mixing parameters
        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        beta_u = beta_dist.sample((batch_size, 1)).to(self.device)
        beta_i = beta_dist.sample((batch_size, 1)).to(self.device)

        # Disordered views
        perm_idx = torch.randperm(batch_size, device=self.device)
        u_dis = u_embeddings[perm_idx]
        pos_dis = pos_embeddings[perm_idx]
        neg_dis = neg_embeddings[perm_idx]

        # Individual mixing
        u_mix = self._mix_embeddings(u_embeddings, u_dis, beta_u)
        pos_mix = self._mix_embeddings(pos_embeddings, pos_dis, beta_i)
        neg_mix = self._mix_embeddings(neg_embeddings, neg_dis, beta_i)

        # Collective mixing
        u_cm = self._collective_mixing(u_embeddings)
        pos_cm = self._collective_mixing(pos_embeddings)

        # Mixed negative BPR
        neg_mix_scores = torch.mul(u_embeddings, neg_mix).sum(dim=1)
        bpr_neg_vec = -F.logsigmoid(pos_scores - neg_mix_scores)

        # Same scalar weighting strategy as WarpRec code
        b_i_scalar = beta_i.mean()
        main_loss_vec = b_i_scalar * bpr_pos_vec + (1.0 - b_i_scalar) * bpr_neg_vec

        # Dual-mixing contrastive loss
        cl_user = self._dual_mixing_cl_loss(
            u_embeddings, u_mix, u_dis, u_cm, beta_u
        )
        cl_item = self._dual_mixing_cl_loss(
            pos_embeddings, pos_mix, pos_dis, pos_cm, beta_i
        )
        cl_loss = self.ssl_lambda * (cl_user + cl_item)

        total_loss = main_loss_vec.mean() + cl_loss + reg_loss

        return total_loss, main_loss_vec

    def predict(self, user_ids):
        """
        Full ranking prediction for AMORe evaluation and fairness wrappers.
        """
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()
        user_embeddings = user_all_embeddings[user_ids]
        item_embeddings = item_all_embeddings[:-1, :]  # remove padding item
        return torch.matmul(user_embeddings, item_embeddings.t())