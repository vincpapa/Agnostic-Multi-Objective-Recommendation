from abc import ABC
import random
from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
from torch_sparse import SparseTensor


class LightGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 args,
                 sparse_train,
                 name="LightGCN",
                 **kwargs
                 ):
        super().__init__()
        random_seed = args.seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = args.dim
        self.n_layers = args.layers
        self.l_w = args.weight_decay
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        row, col = sparse_train.nonzero()
        col = [c + self.num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self.num_users + self.num_items,
                                              self.num_users + self.num_items))
        self.normalize = args.normalize

        self.Gu = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embed_k).to(self.device)
        self.Gi = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embed_k).to(self.device)
        torch.nn.init.normal_(self.Gu.weight, std=0.1)
        torch.nn.init.normal_(self.Gi.weight, std=0.1)

        propagation_network_list = []

        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=self.normalize), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()
        self.myparameters = [self.Gu.weight, self.Gi.weight]

    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.cat((self.Gu.weight.to(self.device), self.Gi.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = torch.mean(torch.stack(all_embeddings, 0), dim=0)
        # all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu, gi

    def compute_xui(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui

    def predict(self, user_id):
        # No torch.no_grad() here: AMORe/MultiFR backpropagate through these scores.
        # Evaluation code already wraps predict() in torch.no_grad().
        gu, gi = self.propagate_embeddings()
        # user_id = Variable(torch.from_numpy(user_id).long(), requires_grad=False).to(self.device)
        # user_emb = self.Gu(user_id)
        # pred = user_emb.mm(self.Gi.weight.t())
        # Raw dot-product scores, as in the other backbones (no sigmoid)
        pred = torch.matmul(gu[user_id].to(self.device), torch.transpose(gi.to(self.device), 0, 1))

        return pred

    def forward(self, user, pos, neg):
        gu, gi = self.propagate_embeddings()
        # user, pos, neg = batch
        xu_pos = self.compute_xui(inputs=(gu[user], gi[pos]))
        xu_neg = self.compute_xui(inputs=(gu[user], gi[neg]))
        maxi = torch.nn.LogSigmoid()(xu_pos - xu_neg)
        mf_loss = -1 * torch.mean(maxi)
        # mf_loss = -1 * torch.sum(maxi)
        reg_loss = self.l_w * (1 / 2) * (torch.norm(gu[user]) ** 2
                                         + torch.norm(gi[pos]) ** 2
                                         + torch.norm(gi[neg]) ** 2) / len(user)
        # reg_loss = self.l_w * (1 / 2) * (torch.norm(gu[user]) ** 2
        #                                  + torch.norm(gi[pos]) ** 2
        #                                  + torch.norm(gi[neg]) ** 2)  # / len(user)
        mf_loss += reg_loss

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # return loss.detach().cpu().numpy()

        return mf_loss

    def custom_forward(self, user, pos, neg):
        gu, gi = self.propagate_embeddings()
        # user, pos, neg = batch
        xu_pos = self.compute_xui(inputs=(gu[user], gi[pos]))
        xu_neg = self.compute_xui(inputs=(gu[user], gi[neg]))
        maxi = -1 * torch.nn.LogSigmoid()(xu_pos - xu_neg)
        b = self.l_w * (1 / 2) * (torch.norm(gu[user], dim=1) ** 2)
        c = self.l_w * (1 / 2) * (torch.norm(gi[pos], dim=1) ** 2)
        d = self.l_w * (1 / 2) * (torch.norm(gi[neg], dim=1) ** 2)

        return torch.mean(maxi + b + c + d), maxi + b + c + d
