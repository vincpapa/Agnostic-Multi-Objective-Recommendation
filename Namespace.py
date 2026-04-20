import torch


class Namespace:
    def __init__(self, settings, experiment):
        self.data = settings['data']
        self.device = torch.device('cuda:' + str(settings['gpu_id']) if torch.cuda.is_available() else 'cpu')
        self.backbone = settings['baseline']
        self.mo_method = settings['wrapper']
        self.mode = experiment['mode']
        self.every = settings['validation_rate']
        self.metric = settings['validation_metric']
        self.batch_size = settings['batch_size']
        self.n_epochs = settings['epochs']
        try:
            self.seed = experiment['seed']
        except KeyError:
            self.seed = 42
        if self.backbone == 'BPRMF':
            self.dim = experiment['dim']
            self.lr = experiment['lr']
            self.weight_decay = experiment['l_2']
        elif self.backbone == 'DirectAU':
            self.dim = experiment['dim']
            self.lr = experiment['lr']
            self.weight_decay = experiment['l_2']
            self.gamma = experiment['gamma']
            self.patience = experiment['patience']
        elif self.backbone == 'LightGCN':
            self.dim = experiment['dim']
            self.lr = experiment['lr']
            self.weight_decay = experiment['l_2']
            self.layers = experiment['layers']
            self.normalize = experiment['normalize']
        elif self.backbone == 'NGCF':
            self.dim = experiment['dim']
            self.lr = experiment['lr']
            self.weight_decay = experiment['l_2']
            self.layers = experiment['layers']
            self.message_dropout = experiment['message_dropout']
            self.node_dropout = experiment['node_dropout']
            self.normalize = experiment['normalize']

        if self.mo_method == 'AMORE_MGDA':
            self.atk = experiment['atk']
            self.type = experiment['g_n']
            self.ranker = experiment['ranker']
            try:
                self.atk_con = experiment['atk']['atk_cons']
            except KeyError:
                pass
            try:
                self.atk_pro = experiment['atk']['atk_prov']
            except KeyError:
                pass
            if self.ranker == 'base':
                self.ablation = experiment['ablation']
        elif self.mo_method == 'multifr':
            self.gamma = experiment['gamma']
            self.temp = experiment['temp']
            self.type = experiment['g_n']
            self.ranker = experiment['ranker']
        elif self.mo_method == 'None':
            self.scale1 = experiment['scale']
        elif self.mo_method in ['AMORE_SCALE', 'AMORE_ABL', 'AMORE_ABL_WOS', 'AMORE_ABL_WOZ', 'AMORE_EPO']:
            try:
                self.atk_con = experiment['atk']['atk_cons']
            except KeyError:
                pass
            try:
                self.atk_pro = experiment['atk']['atk_prov']
            except KeyError:
                pass
            self.ranker = experiment['ranker']
            self.scale1 = experiment['scale']
        elif self.mo_method == 'ADA2FAIR':
            self.weight_lr = experiment['weight_lr']
            self.weight_epochs = experiment['weight_epochs']
            self.topk = experiment['topk']
            self.provider_eta = experiment['provider_eta']
            self.alpha = experiment['alpha']
            self.delta = experiment['delta']

            self.encoder_layers = experiment['encoder_layers']
            self.decoder_layers_pfair = experiment['decoder_layers_pfair']
            self.decoder_layers_ufair = experiment['decoder_layers_ufair']
            self.dropout_prob = experiment['dropout_prob']
            self.encoder_activation = experiment['encoder_activation']




