import pytorch_lightning as pl # 모델 + 학습 로직 관리

from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalMRR

import torch

import torch.nn as nn

import numpy as np



from bert import BERT



class BERT4REC(pl.LightningModule):

    def __init__(self, args):

        super(BERT4REC, self).__init__()



        # 기본 하이퍼파라미터

        self.learning_rate = args.learning_rate         # optimizer learning rate

        self.max_len = args.max_len                      # sequence 길이

        self.hidden_dim = args.hidden_dim               # transformer hidden_dimension

        self.encoder_num = args.encoder_num             # transformer layer 수

        self.head_num = args.head_num                   # multi-head attention head 수

        self.dropout_rate = args.dropout_rate           # FFN dropout

        self.dropout_rate_attn = args.dropout_rate_attn # attention dropout



        # vocab 구성 (0: PAD / 1 ~ itemsize: 실제 item / item_size + 1: MASK token)

        self.vocab_size = args.item_size + 2



        self.initializer_range= args.initializer_range  # weight init 범위

        self.weight_decay = args.weight_decay           # L2 regularization

        self.decay_step = args.decay_step               # Lr scheduler step

        self.gamma = args.gamma                         # lr 감소 비율



        # BERT encoder : sequence -> contextualized embedding 생성

        self.model = BERT(

            vocab_size = self.vocab_size,

            max_len = self.max_len,

            hidden_dim = self.hidden_dim,

            encoder_num = self.encoder_num,

            head_num = self.head_num,

            dropout_rate = self.dropout_rate,

            dropout_rate_attn = self.dropout_rate_attn,

            initializer_range = self.initializer_range

        )



        # output head : (B, T, hidden_dim) -> (B, T, item_size + 1)

        self.out = nn.Linear(self.hidden_dim, args.item_size + 1)



        self.batch_size = args.batch_size



        # loss (ignore_index = 0 -> PAD는 loss 계산에서 제외)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)



        # 평가 metric

        self.HR   = RetrievalHitRate(top_k=10)

        self.NDCG = RetrievalNormalizedDCG(top_k=10)

        self.MRR  = RetrievalMRR()     



    def training_step(self, batch, batch_idx):

        """

        BERT4REC 학습 단계



        batch:

            seq : input sequence (mask 포함)

            labels : 정답 item (mask 위치만 값 있음, 나머지는 0)

        """



        seq, labels = batch



        logits = self.model(seq)  # (B, T, hidden)

        preds = self.out(logits)  # (B, T, item_size + 1)



        # CrossEntropyLoss 입력 형태 맞추기 위해 transpose

        loss = self.criterion(preds.transpose(1, 2), labels)



        # 로그 기록

        self.log("train_loss", loss,

                 on_step=True, on_epoch=True,

                 prog_bar=True, logger=True)

    

        return loss

    

    def validation_step(self, batch, batch_idx):

        """

        validation 단계 (ranking 기반 평가)



        batch:

            seq : input sequence

            candidate : [정답 + negative items] (B, 101)

            labels : 정답 위치 (one-hot 형태)

        """

        seq, candidates, labels = batch



        logits = self.model(seq)

        preds = self.out(logits)



        # 마지막 위치 prediction : BERT4REC은 마지막 mask 위치만 평가

        preds = preds[:, -1, :] # (B, item_size + 1)

        

        # 정답 item id

        targets = candidates[:, 0]



        # classification loss (참고용)

        loss = self.criterion(preds, targets)



        # candidate subset score 추출

        recs = torch.gather(preds, 1, candidates)



        # metrics 계산용 index 생성

        steps = batch_idx * self.batch_size

        indexes = torch.arange(

            steps, steps + seq.size(0),

            dtype=torch.long,

            device=seq.device

        ).unsqueeze(1).repeat(1, 101)



                # ===== logging =====

        self.log("val_loss", loss,

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)



        self.log("HR_val",

                 self.HR(recs, labels, indexes),

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)



        self.log("NDCG_val",

                 self.NDCG(recs, labels, indexes),

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        

        self.log("MRR_val",

                 self.MRR(recs, labels, indexes),

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        

    def test_step(self, batch, batch_idx):

        """

        test 단계 (validation과 동일 구조)

        """

        seq, candidates, labels = batch



        logits = self.model(seq)

        preds  = self.out(logits)



        preds   = preds[:, -1, :]

        targets = candidates[:, 0]

        loss    = self.criterion(preds, targets)



        recs = torch.gather(preds, 1, candidates)



        steps   = batch_idx * self.batch_size

        indexes = torch.arange(

            steps, steps + seq.size(0),

            dtype=torch.long,

            device=seq.device

        ).unsqueeze(1).repeat(1, 101)



        self.log("test_loss", loss,

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)



        self.log("HR_test",

                 self.HR(recs, labels, indexes),

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)



        self.log("NDCG_test",

                 self.NDCG(recs, labels, indexes),

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        

        self.log("MRR_test",

                 self.MRR(recs, labels, indexes),

                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        """

        inference (추천 결과 생성)



        입력 : seq

        출력 : top-10 item index

        """

        seq = batch



        logits = self.model(seq)

        preds = self.out(logits)



        preds = preds[:, -1, :] # 마지막 step 기준 추천



        # top-10 추천

        indexes, _ = torch.topk(preds, 10)



        return indexes.cpu().numpy()

    

    def configure_optimizers(self):

        """

        optimizer + scheduler 설정

        """

        # weight decay 적응 분리

        no_decay = ['bias', 'LayerNorm.weight']



        params = [

            {

                'params' : [p for n, p in self.named_parameters()

                            if not any(nd in n for nd in no_decay)],

                'weight_decay' : self.weight_decay

            },

            {

                'params' : [p for n, p in self.named_parameters()

                            if any(nd in n for nd in no_decay)],

                'weight_decay' : 0.0

            }

        ]
