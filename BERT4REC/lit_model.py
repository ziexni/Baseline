import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalMRR

# 참고: BERT 클래스는 외부에서 정의되어 있다고 가정합니다.
# from bert import BERT 

class BERT4REC(pl.LightningModule):
    def __init__(self, args):
        super(BERT4REC, self).__init__()

        # 하이퍼파라미터 설정
        self.save_hyperparameters(args)
        self.learning_rate = args.learning_rate
        self.item_size = args.item_size
        self.batch_size = args.batch_size
        
        # vocab: 0(PAD), 1~item_size(Items), item_size+1(MASK)
        self.vocab_size = args.item_size + 2

        # BERT 모델 초기화
        # self.model = BERT(...) 
        # 예시 구조 (실제 구현체에 맞춰 수정 필요)
        self.model = BERT(
            vocab_size=self.vocab_size,
            max_len=args.max_len,
            hidden_dim=args.hidden_dim,
            encoder_num=args.encoder_num,
            head_num=args.head_num,
            dropout_rate=args.dropout_rate,
            dropout_rate_attn=args.dropout_rate_attn,
            initializer_range=args.initializer_range
        )

        # Output Layer: (B, T, Hidden) -> (B, T, item_size + 1) 
        # 0번(PAD)을 포함한 아이템 점수 계산
        self.out = nn.Linear(args.hidden_dim, args.item_size + 1)

        # Loss: PAD(0)는 제외
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 평가 Metric (Top-10 기준)
        # Retrieval 계열 메트릭은 (preds, target, indexes) 형태를 받습니다.
        self.hr_10 = RetrievalHitRate(top_k=10)
        self.ndcg_10 = RetrievalNormalizedDCG(top_k=10)
        self.mrr = RetrievalMRR()

    def forward(self, seq):
        logits = self.model(seq)
        return self.out(logits)

    def training_step(self, batch, batch_idx):
        seq, labels = batch # labels: (B, T) - MASK된 위치만 아이템ID, 나머지는 0
        
        preds = self(seq) # (B, T, item_size + 1)
        
        # CrossEntropy를 위해 shape 변경: (B, C, T)
        loss = self.criterion(preds.transpose(1, 2), labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx, phase):
        """
        Validation과 Test에서 공통으로 사용하는 101 후보군 평가 로직
        batch 구성:
            - seq: 입력 시퀀스 (B, T)
            - candidates: [정답 아이템, Neg 1, ..., Neg 100] (B, 101)
            - labels: [1, 0, ..., 0] (B, 101) - 첫 번째가 정답임을 나타내는 binary
        """
        seq, candidates, labels = batch
        batch_size = seq.size(0)

        # 1. 모델 예측 (마지막 타임스텝의 결과만 사용 - Leave-one-out 기반)
        logits = self(seq)
        preds_at_last = logits[:, -1, :] # (B, item_size + 1)

        # 2. 101개 후보군에 대해서만 점수 추출 (Candidate Ordering 유지)
        # candidates의 각 행에 해당하는 아이템의 확률값만 가져옴
        candidate_scores = torch.gather(preds_at_last, 1, candidates) # (B, 101)

        # 3. Retrieval Metrics를 위한 indexes 생성
        # 각 유저(샘플)마다 독립된 랭킹을 계산하기 위해 고유 index 부여
        step_offset = batch_idx * self.batch_size
        indexes = torch.arange(
            step_offset, step_offset + batch_size, 
            device=self.device
        ).unsqueeze(1).expand_as(candidate_scores)

        # 4. Metric 계산
        # labels는 해당 아이템이 정답인지 여부(bool/int)
        hr = self.hr_10(candidate_scores, labels, indexes=indexes)
        ndcg = self.ndcg_10(candidate_scores, labels, indexes=indexes)
        mrr = self.mrr(candidate_scores, labels, indexes=indexes)

        # 로그 기록
        self.log(f"{phase}_HR@10", hr, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_NDCG@10", ndcg, on_epoch=True, prog_bar=True)
        self.log(f"{phase}_MRR", mrr, on_epoch=True, prog_bar=True)

        return hr

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # Weight Decay 적용 범위 설정
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.decay_step, gamma=self.hparams.gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--max_len", type=int, default=50)
        parser.add_argument("--encoder_num", type=int, default=2)
        parser.add_argument("--head_num", type=int, default=4)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate_attn", type=float, default=0.1)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--decay_step", type=int, default=25)
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument("--item_size", type=int, required=True)
        parser.add_argument("--batch_size", type=int, default=128)
        return parser
