import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert import BERT


class BERT4REC(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # ... 기존 코드 동일 ...

        self.model = BERT(...)

        # ✅ Weight tying: output projection = input embedding 가중치 공유
        # self.out = nn.Linear(hidden_dim, item_size+1) 대신:
        self.out_bias = nn.Parameter(torch.zeros(self.item_size + 1))
        # forward에서 self.model.embedding.token_embeddings.weight 재사용

        self.bce = nn.BCEWithLogitsLoss()

    def _get_scores(self, hidden):
        """
        Weight tying: hidden @ token_emb.T + bias
        hidden: (B, hidden_dim) 또는 (B, T, hidden_dim)
        """
        # token_emb.weight: (vocab_size, hidden_dim)
        # item 범위만 사용: [0 : item_size+1]
        emb_weight = self.model.embedding.token_embeddings.weight[:self.item_size + 1]
        # (B, T, hidden) @ (hidden, item_size+1) → (B, T, item_size+1)
        return torch.matmul(hidden, emb_weight.T) + self.out_bias

    def training_step(self, batch, batch_idx):
        seq, pos, neg = batch

        hidden = self.model(seq)           # (B, T, hidden)
        preds  = self._get_scores(hidden)  # (B, T, item_size+1)

        indices   = torch.where(pos != 0)
        pos_score = preds[indices[0], indices[1], pos[indices]]
        neg_score = preds[indices[0], indices[1], neg[indices]]

        loss = (
            self.bce(pos_score, torch.ones_like(pos_score)) +
            self.bce(neg_score, torch.zeros_like(neg_score))
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, candidates, _ = batch
        hidden = self.model(seq)
        preds  = self._get_scores(hidden)[:, -1, :]   # (B, item_size+1)
        scores = torch.gather(preds, 1, candidates)

        hr, ndcg, mrr = self.evaluate_batch(scores)
        self.log("HR_val",   hr,   prog_bar=True)
        self.log("NDCG_val", ndcg, prog_bar=True)
        self.log("MRR_val",  mrr,  prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq, candidates, _ = batch
        hidden = self.model(seq)
        preds  = self._get_scores(hidden)[:, -1, :]
        scores = torch.gather(preds, 1, candidates)

        hr, ndcg, mrr = self.evaluate_batch(scores)
        self.log("HR_test",   hr)
        self.log("NDCG_test", ndcg)
        self.log("MRR_test",  mrr)

    # configure_optimizers 이하 동일


    # -------------------------
    # Optimizer
    # -------------------------
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        params = [
            {
                'params': [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--learning_rate",     type=float, default=1e-3)
        parser.add_argument("--hidden_dim",        type=int,   default=128)
        parser.add_argument("--encoder_num",       type=int,   default=2)
        parser.add_argument("--head_num",          type=int,   default=4)
        parser.add_argument("--dropout_rate",      type=float, default=0.1)
        parser.add_argument("--dropout_rate_attn", type=float, default=0.1)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--weight_decay",      type=float, default=0.01)
        parser.add_argument("--decay_step",        type=int,   default=25)
        parser.add_argument("--gamma",             type=float, default=0.1)
        return parser
