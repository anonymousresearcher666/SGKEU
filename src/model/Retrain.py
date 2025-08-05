from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_

from src.model.kge_models.TransE import TransE
from src.utilities.utilities import *


class Retrain(TransE):
    def __init__(self, args, kg) -> None:
        super(Retrain, self).__init__(args, kg)
        self.args = args
        self.kg = kg
        self.ent_embeddings = nn.Embedding(self.kg.ent_num, self.args.emb_dim).to(self.args.device).float()
        self.rel_embeddings = nn.Embedding(self.kg.rel_num, self.args.emb_dim).to(self.args.device).float()
        xavier_normal_(self.ent_embeddings.weight)
        xavier_normal_(self.rel_embeddings.weight)
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")

    def embedding(self):
        return self.ent_embeddings.weight, self.rel_embeddings.weight

    def ent_norm(self, e):
        return F.normalize(e, 2, -1)

    def rel_norm(self, r):
        return F.normalize(r, 2, -1)

    def score_fun(self, h, r, t):
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t = self.ent_norm(t)
        return torch.norm(h + r - t, 1, -1)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def margin_loss(self, head, relation, tail, label=None):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.margin_loss(head, relation, tail, label) / head.size(0)
        return loss

    def predict(self, head, relation):
        ent_embeddings, rel_embeddings = self.embedding()
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t_all = ent_embeddings
        h = self.ent_norm(h)
        r = self.rel_norm(r)
        t_all = self.ent_norm(t_all)

        """ h + r - t """
        pred_t = h + r
        score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_all, p=1, dim=2)
        score = torch.sigmoid(score)
        return score
