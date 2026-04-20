import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_layers, output_dim, dropout=0.0, activation="relu", final_activation=False):
    layers = []
    prev = input_dim
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "leakyrelu": nn.LeakyReLU,
    }
    act_cls = act_map.get(str(activation).lower(), nn.ReLU)

    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    layers.append(nn.Linear(prev, output_dim))
    if final_activation:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class Ada2FairModel(nn.Module):
    """
    Wrapper full-style Ada2Fair for AMORe.

    Keeps:
    - backbone model unchanged
    - weight generator (encoder + two decoders)
    - alternating optimization
    - weighted recommendation loss for stage II

    Assumes:
    - provider_ids: numpy array of shape [num_items], each item mapped to provider/group
    """

    def __init__(self, backbone, train_matrix, train_user_list, provider_ids, args):
        super().__init__()
        self.backbone = backbone
        self.device = args.device

        self.num_users, self.num_items = train_matrix.shape
        self.train_user_list = train_user_list

        provider_ids = np.asarray(provider_ids)
        assert provider_ids.shape[0] == self.num_items
        self.provider_ids = torch.tensor(provider_ids, dtype=torch.long, device=self.device)
        self.num_providers = int(provider_ids.max()) + 1

        dense_rating = torch.tensor(
            train_matrix.toarray(),
            dtype=torch.float32,
            device=self.device
        )
        self.register_buffer("rating_matrix", dense_rating)

        self.alpha = getattr(args, "ada2fair_alpha", 0.5)
        self.topk = getattr(args, "ada2fair_topk", 20)
        self.weight_epochs = getattr(args, "weight_epochs", 1)
        self.provider_eta = getattr(args, "provider_eta", 1.0)
        self.delta = getattr(args, "delta", 1e-8)

        self.encoder_layers = getattr(args, "encoder_layers", [256, 128])
        self.decoder_layers_pfair = getattr(args, "decoder_layers_pfair", [128])
        self.decoder_layers_ufair = getattr(args, "decoder_layers_ufair", [128])
        self.dropout_prob = getattr(args, "dropout_prob", 0.1)
        self.activation = getattr(args, "encoder_activation", "relu")

        # Encoder maps user interaction vector -> item-wise adaptive weights
        self.encoder = build_mlp(
            input_dim=self.num_items,
            hidden_layers=self.encoder_layers,
            output_dim=self.num_items,
            dropout=self.dropout_prob,
            activation=self.activation,
            final_activation=False,
        )

        # Decoders reconstruct provider-side and user-side targets
        self.decoder_pfair = build_mlp(
            input_dim=self.num_items,
            hidden_layers=self.decoder_layers_pfair,
            output_dim=self.num_items,
            dropout=self.dropout_prob,
            activation=self.activation,
            final_activation=False,
        )

        self.decoder_ufair = build_mlp(
            input_dim=self.num_items,
            hidden_layers=self.decoder_layers_ufair,
            output_dim=self.num_items,
            dropout=self.dropout_prob,
            activation=self.activation,
            final_activation=False,
        )

        self.loss_func = nn.MSELoss()

        self.provider_fairness_weight = None   # [num_items]
        self.user_fairness_weight = None       # [num_users]
        self.fairness_weight_matrix = None     # [num_users, num_items]

    def backbone_parameters(self):
        return self.backbone.parameters()

    def weight_parameters(self):
        return list(self.encoder.parameters()) + \
               list(self.decoder_pfair.parameters()) + \
               list(self.decoder_ufair.parameters())

    def forward(self, user, pos, neg):
        return self.backbone(user, pos, neg)

    def predict(self, user_ids):
        return self.backbone.predict(user_ids)

    def custom_forward(self, user, pos, neg):
        return self.backbone.custom_forward(user, pos, neg)

    @torch.no_grad()
    def update_fairness_targets(self):
        """
        Build provider-side and customer-side fairness targets
        from current backbone predictions.
        """
        all_users = torch.arange(self.num_users, dtype=torch.long, device=self.device)
        scores = self.predict(all_users).detach()

        # mask seen items
        seen_mask = self.rating_matrix > 0
        scores = scores.masked_fill(seen_mask, -1e10)

        topk_scores, topk_items = torch.topk(scores, k=self.topk, dim=1)

        # rank discounts
        discounts = 1.0 / torch.log2(
            torch.arange(2, self.topk + 2, device=self.device, dtype=torch.float32)
        )
        discounts = discounts.unsqueeze(0).repeat(self.num_users, 1)

        # item exposure
        item_exposure = torch.zeros(self.num_items, device=self.device)
        item_exposure.scatter_add_(
            0,
            topk_items.reshape(-1),
            discounts.reshape(-1)
        )

        # aggregate by provider
        provider_exposure = torch.zeros(self.num_providers, device=self.device)
        provider_exposure.scatter_add_(
            0,
            self.provider_ids[topk_items.reshape(-1)],
            discounts.reshape(-1)
        )

        provider_item_count = torch.bincount(
            self.provider_ids,
            minlength=self.num_providers
        ).float().to(self.device)

        # same spirit as Ada2Fair trainer: normalize by provider item count
        provider_avg_exposure = provider_exposure / torch.clamp(provider_item_count, min=1.0)

        # disadvantaged providers get larger weights
        provider_fairness = torch.pow(
            torch.clamp(provider_avg_exposure + self.delta, min=self.delta),
            -self.provider_eta
        )

        provider_fairness = provider_fairness / provider_fairness.mean()

        # lift provider target back to item space
        item_provider_weight = provider_fairness[self.provider_ids]
        item_provider_weight = item_provider_weight / item_provider_weight.mean()

        # less active users get higher user-side fairness targets
        user_hist_len = torch.tensor(
            [max(len(x), 1) for x in self.train_user_list],
            dtype=torch.float32,
            device=self.device
        )
        user_fairness = 1.0 / user_hist_len
        user_fairness = user_fairness / user_fairness.mean()

        self.provider_fairness_weight = item_provider_weight.detach()
        self.user_fairness_weight = user_fairness.detach()

    def weight_loss(self, user_ids):
        """
        Stage I loss: learn adaptive user-item weights from
        provider-side and customer-side fairness targets.
        """
        assert self.provider_fairness_weight is not None
        assert self.user_fairness_weight is not None

        rating_matrix = self.rating_matrix[user_ids]  # [B, I]

        h_encode = self.encoder(rating_matrix)
        h_encode = h_encode * rating_matrix

        h_encode_pfair = self.decoder_pfair(h_encode)
        h_encode_ufair = self.decoder_ufair(h_encode)

        target_pfair = self.decoder_pfair(
            (self.provider_fairness_weight.unsqueeze(0).repeat(user_ids.size(0), 1) * rating_matrix).float()
        )

        target_ufair = self.decoder_ufair(
            (
                self.user_fairness_weight[user_ids]
                .unsqueeze(1)
                .repeat(1, self.num_items)
                * rating_matrix
            ).float()
        )

        loss_pfair = self.loss_func(h_encode_pfair, target_pfair)
        loss_ufair = self.loss_func(h_encode_ufair, target_ufair)

        return loss_pfair, loss_ufair

    @torch.no_grad()
    def export_fairness_weight_matrix(self, batch_size=256):
        """
        Stage I -> Stage II bridge:
        generate item-wise adaptive weights for all users.
        """
        out = torch.zeros(self.num_users, self.num_items, device=self.device)
        for start in range(0, self.num_users, batch_size):
            end = min(start + batch_size, self.num_users)
            batch = self.rating_matrix[start:end]
            out[start:end] = self.encoder(batch.float())

        # positivity + normalization
        out = F.softplus(out)
        row_mean = torch.clamp(out.mean(dim=1, keepdim=True), min=1e-8)
        out = out / row_mean
        self.fairness_weight_matrix = out.detach()

    def weighted_recommendation_loss(self, user, pos, neg):
        """
        Stage II: weighted recommendation loss.
        """
        mean_loss, sample_loss = self.custom_forward(user, pos, neg)
        if self.fairness_weight_matrix is None:
            return mean_loss

        ipw = self.fairness_weight_matrix[user, pos].detach()
        return torch.mean(sample_loss * ipw)