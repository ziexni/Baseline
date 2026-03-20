import torch
import torch.nn as nn
from functools import partial # 함수의 일부 인자를 미리 고정한 새로운 함수 생성 기능
# from recbole.model.abstract_recommender import SequentialRecommender # Sequential 추천 모델 기본 클래스
from recbole.model.layers import TransformerEncoder # 순차 관계 학습 인코더
from recbole.model.layers import FeatureSeqEmbLayer # feature => embedding
from recbole.model.layers import VanillaAttention # 중요한 아이템에 더 높은 가중치
from recbole.model.loss import BPRLoss # Bayesian Personalized Ranking
from einops.layers.torch import Rearrange, Reduce # 텐서 차원 변형 라이브러리
import argparse
import pickle # 파일 저장/로드
import numpy as np
import pandas as pd
from time import time
import torch.nn.functional as F
import ast

# 입력값이 Tuple이면 그대로 사용, 아니면 (x, x) 형태의 Tuple로 변환
pair = lambda x: x if isinstance(x, tuple) else (x, x)

# LayerNorm => Attention(fn) => Residual
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn # 입력 받은 function
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x # + residual connection

# Feed-Forward Network
def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor) # 차원 확장
    return nn.Sequential( # 여러 Layer 순차 실행
        dense(dim, inner_dim), # Fully connected layer
        nn.GELU(), # Gaussian Error Linear Unit
        nn.Dropout(dropout),
        dense(inner_dim, dim), # 차원 축소
        nn.Dropout(dropout)
    )

'''
class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(300, 150),
            nn.LeakyReLU(inplace=True),
            nn.Linear(150, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 64),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.model(x)

        return x
    
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 150),
            nn.LeakyReLU(inplace=True),
            nn.Linear(150, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 128),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x
    
class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 150),
            nn.LeakyReLU(inplace=True),
            nn.Linear(150, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x
'''    

# Model
class MMMLP(nn.Module):
    r"""
    MLP-Mixer 기반 멀티모달 순차 추천 모델,
    ID / Text / Visual 3가지 모달리티를 각각 MLP-Mixer로 인코딩 후 fusion
    """

    def __init__(self, args, intr_dataset, title_dataset, item_dataset, category2id):
        super(MMMLP, self).__init__()

        # feature dataset: item_id 1-indexed
        item_dataset = item_dataset.copy()
        item_dataset['item_id'] = item_dataset['item_id'] + 1

        # hyperparameters
        self.n_layers = args.num_blocks # config['n_layers']
        self.hidden_size = args.hidden_units # config['hidden_size'] same as embedding size
        self.hidden_dropout_prob = args.dropout_rate # config['hidden_dropout_prob']
        self.hidden_act = 'gelu' # config['hidden_act']
        self.layer_norm_eps = 1e-8 # config['layer_norm_eps']
        self.max_seq_length = args.maxlen
        self.device = args.device

        self.title_dataset = title_dataset
        self.intr_dataset = intr_dataset
        self.item_dataset = item_dataset
        self.category2id = category2id

        self.title_dim = title_dataset.shape[1]

        self.item_to_category = dict(
            zip(self.item_dataset['item_id'], self.item_dataset['category_id'])
        )

        # projections
        self.t_proj = nn.Linear(384, self.hidden_size) # 384 => 100
        self.v_proj = nn.Linear(2048, self.hidden_size) # 2048 => 100

        self.n_items = intr_dataset['item_id'].max() + 2
        n_categories = len(category2id)
        expansion_factor = 4
        chan_first = partial(nn.Conv1d, kernel_size=1) # kernel size 고정
        chan_last = nn.Linear

        # 4개의 feature을 concat한 후, 다시 hidden_size로 축소할 용도
        self.concat_layer_f = nn.Linear(self.hidden_size*4, self.hidden_size)

        self.initializer_range = 0.02
        self.loss_type = 'BPR'

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # self.category_embedding = nn.EmbeddingBag(n_categories, self.hidden_size, mode='mean')
        self.c_proj = nn.Linear(n_categories, self.hidden_size)

        # Item MLP-Mixer
        # Tokem Mixing Layer: 시퀀스 토큰들 간 정보(아이템 순서 간 관계)를 MLP로 섞는 작업 (transformer => MLP)
        self.tokenMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        # Channel Mixing: Feature 차원들 간 관계를 MLP로 섞는 작업
        self.channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Text MLP-Mixer
        self.t_tokenMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.t_channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.t_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Visual MLP-MIxer
        self.v_tokenMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.v_channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.v_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Category MLP-Mixer
        self.c_tokenMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.c_channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.c_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Fusion LayerNorm + Dropout
        self.LayerNormFeature = nn.LayerNorm(4*self.hidden_size, eps=self.layer_norm_eps) # 3
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.batch_size = args.batch_size # 한 batch 데이터 수
        self.batch_num = max(self.batch_size // 4, 1) # batch 개수

        # Process the image data here
        print(repr(item_dataset['video_feature'].iloc[0]))

        self.image_feature = item_dataset['video_feature'].apply(
            lambda x: np.array(x, dtype=np.float32)
        )
        self.image_feature = torch.tensor(
            np.stack(self.image_feature.values),  # tolist() 보다 np.stack이 더 안전
            dtype=torch.float32
        ).to(self.device)

        self.image_feature_dim = self.image_feature.shape[1]  

        # Loss
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        
        # parameters initialization
        self.apply(self._init_weights)
        self._build_feature_table()


    # Feature Table 사전 빌드 (룩업 속도 최적화)
    def _build_feature_table(self):
        n = self.n_items

        # Visual
        item_ids = self.item_dataset['item_id'].values 
        
        v_table = torch.zeros(n, self.image_feature_dim).to(self.device)
        v_feats = self.image_feature  # 이미 [n_items, feat_dim] 텐서로 존재
        valid_mask = (item_ids < n)
        v_table[item_ids[valid_mask]] = v_feats[torch.tensor(valid_mask)]
        self.v_table = v_table.to(self.device)

        # Text 
        t_table = torch.zeros(n, self.title_dim).to(self.device)
        title_tensor = torch.tensor(self.title_dataset, dtype=torch.float32).to(self.device)  # [n_items, title_dim]
        # title_dataset[i] → item_id = i+1
        title_ids = np.arange(1, len(self.title_dataset) + 1)  # 0-indexed → 1-indexed
        valid_mask = (title_ids < n)
        t_table[title_ids[valid_mask]] = title_tensor[valid_mask]
        self.t_table = t_table.to(self.device)

        # Category
        n_cat = len(self.category2id)
        c_table = torch.zeros(n, n_cat).to(self.device)
        for item_id, cat_ids in self.item_to_category.items():
            idx = item_id 
            if idx < n:
                cat_ids_flat = np.array(cat_ids).flatten().astype(int)
                valid_cats = cat_ids_flat[(cat_ids_flat >= 0) & (cat_ids_flat < n_cat)]
                c_table[idx, valid_cats] = 1.0
        self.c_table = c_table.to(self.device)

    # weight 초기화
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range) # 정규분포로 값 초기화
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    # Modality Mixer
    def vMLPMixer(self, item_seq):
        # Visual feature MLP-Mixer [batch, seq, hidden]
        out = self.v_table[item_seq]   
        out = self.v_proj(out)
        out = self.v_LayerNorm(out)
        for _ in range(self.n_layers):
            out = self.v_tokenMixer(out)
            out = self.v_channelMixer(out)
        return out
    
    def cMLPMixer(self, item_seq):
        # Category feature MLP-Mixer [batch, seq, hidden]
        out = self.c_table[item_seq]         
        out = self.c_proj(out)
        out = self.c_LayerNorm(out)
        for _ in range(self.n_layers):
            out = self.c_tokenMixer(out)
            out = self.c_channelMixer(out)
        return out
    
    def tMLPMixer(self, item_seq):
        # Text feature MLP-Mixer [batch, seq, hidden]
        out = self.t_table[item_seq]          
        out = self.t_proj(out)
        out = self.t_LayerNorm(out)
        for _ in range(self.n_layers):
            out = self.t_tokenMixer(out)
            out = self.t_channelMixer(out)
        return out

    # Item 멀티모달 임베딩 (평가/예측용)
    def get_item_multimodal(self, item_ids):

        id_emb = self.item_embedding(item_ids)

        t = self.t_proj(self.t_table[item_ids])
        v = self.v_proj(self.v_table[item_ids])
        c = self.c_proj(self.c_table[item_ids])

        item_vec = torch.cat([id_emb, t, v, c], dim=-1)
        item_vec = self.concat_layer_f(item_vec)

        return item_vec
    
    # Batch 예측 (평가 전용)
    def predict_batch(self, seqs, item_idxs):
        """
        seqs:      [batch, maxlen] numpy
        item_idxs: [batch, n_candidates] numpy
        returns:   [batch, n_candidates] numpy
        """
        seq = torch.LongTensor(seqs).to(self.device)
        seq_len = (seq != 0).sum(dim=1)
        seq_output = self.forward(seq, seq_len)  # [batch, hidden]

        item_idx = torch.LongTensor(item_idxs).to(self.device)  # [batch, n_cand]
        item_emb = self.get_item_multimodal(item_idx)        # [batch, n_cand, hidden]

        # einsum으로 배치 내적
        logits = torch.einsum('bh,bnh->bn', seq_output, item_emb)  # [batch, n_cand]
        return logits.detach().cpu().numpy()

    # Forward
    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq) 
        mixer_output = self.LayerNorm(item_emb) # 초기 정규화

        for _ in range(self.n_layers):
            mixer_output = self.tokenMixer(mixer_output)
            mixer_output = self.channelMixer(mixer_output)

        # fusion => 멀티모달 아이템 표현 생성 
        fusemixer_output = torch.cat([
            mixer_output, # ID embedding
            self.tMLPMixer(item_seq), # title
            self.vMLPMixer(item_seq), # image
            self.cMLPMixer(item_seq) # category
        ], dim=-1)
        
        # 최종 사용자 표현 벡터 생성 단계
        fusemixer_output = self.LayerNormFeature(fusemixer_output)
        fusemixer_output = self.dropout(fusemixer_output)
        fusemixer_output = self.concat_layer_f(fusemixer_output) # 차원 축소

        # 마지막 유효 아이템 벡터 추출
        last_idx = item_seq_len - 1
        batch_size, seq_len, hidden_size = fusemixer_output.shape
        seq_output = fusemixer_output[torch.arange(batch_size), last_idx, :]
        seq_output = self.LayerNorm(seq_output)

        return seq_output

    # 모델 학습 시 손실(loss) 계산 => prediction vs 정답
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len) # (batch, hidden_size)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            
            # 마지막 아이템만 선택
            pos_last = pos_items[:, -1]  # [batch]
            neg_last = neg_items[:, -1]  # [batch]
            
            pos_items_emb = self.get_item_multimodal(pos_last)
            neg_items_emb = self.get_item_multimodal(neg_last)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            
            return loss
        
    def predict(self, user_ids, seq, item_idx):
        seq = torch.LongTensor(seq).to(self.device)

        seq_len = (seq != 0).sum(dim=1)

        seq_output = self.forward(seq, seq_len)

        item_idx = torch.LongTensor(item_idx).to(self.device)

        item_emb = self.item_embedding(item_idx)

        logits = torch.matmul(seq_output, item_emb.t())

        return logits.detach().cpu().numpy()

