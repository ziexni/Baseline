import pytorch_lightning as pl
import torch
import torch.nn as nn

from bert import BERT


class BERT4REC(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.learning_rate = args.learning_rate
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.encoder_num = args.encoder_num
        self.head_num = args.head_num
        self.dropout_rate = args.dropout_rate
        self.dropout_rate_attn = args.dropout_rate_attn

        self.vocab_size = args.item_size + 2
        self.item_size = args.item_size

        self.weight_decay = args.weight_decay
        self.decay_step = args.decay_step
        self.gamma = args.gamma
        self.batch_size = args.batch_size

        # BERT encoder
        self.model = BERT(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            hidden_dim=self.hidden_dim,
            encoder_num=self.encoder_num,
            head_num=self.head_num,
            dropout_rate=self.dropout_rate,
            dropout_rate_attn=self.dropout_rate_attn,
            initializer_range=args.initializer_range
        )

        # output head
        self.out = nn.Linear(self.hidden_dim, self.item_size + 1)

    # -------------------------
    # LOSS (PAIRWISE RANKING)
    # -------------------------
    def training_step(self, batch, batch_idx):
        seq, candidates, labels = batch

        logits = self.model(seq)
        preds = self.out(logits)[:, -1, :]  # (B, item_size+1)

        scores = torch.gather(preds, 1, candidates)  # (B, 101)

        pos = scores[:, 0]
        neg = scores[:, 1:]

        loss = -torch.log(torch.sigmoid(pos.unsqueeze(1) - neg)).mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------
    # EVAL METRICS (SAME AS AMPREC)
    # -------------------------
    @staticmethod
    def compute_metrics(scores):
        # scores: (B, 101), index 0 = positive
        rank = scores.argsort(dim=1, descending=True)

        pos_rank = (rank == 0).nonzero(as_tuple=True)[1] + 1  # 1-based rank

        hr = (pos_rank <= 10).float().mean()
        mrr = (1.0 / pos_rank.float()).mean()
        ndcg = (1.0 / torch.log2(pos_rank.float() + 1)).mean()

        return hr, ndcg, mrr

    def validation_step(self, batch, batch_idx):
        seq, candidates, labels = batch

        logits = self.model(seq)
        preds = self.out(logits)[:, -1, :]

        scores = torch.gather(preds, 1, candidates)

        hr, ndcg, mrr = self.compute_metrics(scores)

        self.log("HR_val", hr, prog_bar=True)
        self.log("NDCG_val", ndcg, prog_bar=True)
        self.log("MRR_val", mrr, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq, candidates, labels = batch

        logits = self.model(seq)
        preds = self.out(logits)[:, -1, :]

        scores = torch.gather(preds, 1, candidates)

        hr, ndcg, mrr = self.compute_metrics(scores)

        self.log("HR_test", hr)
        self.log("NDCG_test", ndcg)
        self.log("MRR_test", mrr)

    # -------------------------
    # INFERENCE
    # -------------------------
    def predict_step(self, batch, batch_idx):
        seq = batch

        logits = self.model(seq)
        preds = self.out(logits)[:, -1, :]

        _, topk = torch.topk(preds, 10)

        return topk.cpu().numpy()

    # -------------------------
    # OPTIMIZER
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
            optimizer,
            step_size=self.decay_step,
            gamma=self.gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    
    @staticmethod
    def add_to_argparse(parser):
        """
        CLI argument 정의
        """
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
