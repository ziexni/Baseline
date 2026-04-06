"""
data.py
MovieLens -> MicroVideo 데이터셋으로 교체
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset # Pytorch에게 내 데이터를 정의해 주기 위한 클래스
from collections import defaultdict
import pickle

def get_data(interaction_path):
    """
    Interaction file load 
    Returns :
        user_train, user_valid, user_test : [(item_id)]
        usernum, itemnum
    """
    with open('bigMatrix.pkl', 'rb') as f:
        df = pickle.load(f)

    df['user_id'] = df['user_id'] + 1
    df['video_id'] = df['video_id'] + 1
    df = df.sort_values(by=['user_id', 'timestamp'], kind='mergesort').reset_index(drop=True)

    usernum = df['user_id'].max()
    itemnum = df['video_id'].max()

    User = defaultdict(list)
    for u, i in zip(df['user_id'], df['video_id']):
       User[u].append(int(i))

    user_train, user_valid, user_test = {}, {}, {}
    for user, seq in User.items():
        n = len(seq)
        if n < 3: # 상호작용이 3개 미만인 경우
            user_train[user] = seq
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user] = [seq[-1]]

    print(f"[Split] train users: {len(user_train)}, "
          f"valid: {sum(1 for v in user_valid.values() if v)}, "
          f"test: {sum(1 for v in user_test.values() if v)}")

    return user_train, user_valid, user_test, usernum, itemnum

class MicroVideoDataset(Dataset):
    """
    BERT4REC Cloze task Dataset
    
    Pytorch Dataset을 상속해서 DataLoader가 사용할 수 있도록 만든 것

    [ 토큰 규칙 (Sequence 안에 들어가는 값 의미) ]
    PAD = 0 # padding 토큰
    item = 1 ~ itemnum # 실제 아이템 ID (1-indexed)
    MASK = itemnum + 1 # prediction을 위해 가리는 토큰

    [ mode ]
    1. 'train'
    - sequence 내부에서 랜덤하게 MASK 적용
    - 모델이 가려진 아이템을 맞추도록 학습
    - 반환값 : (seq, labels)
    2. 'valid' / 'test'
    - 마지막 아이템을 MASK 처리
    - 모델이 다음 아이템을 맞추도록 평가
    - 반환값 : (seq, candidates, labels)
    - candidates = [정답 아이템] + 100개의 negative 샘플
    """
    def __init__(self, user_train, user_valid, user_test,
                 itemnum, maxlen, mask_prob=0.2, neg_sample_size=100, mode='train', usernum=0):
        
        # 데이터 저장
        self.user_train = user_train # {user: sequence} 형태의 train 데이터
        self.user_valid = user_valid # validation 정답
        self.user_test = user_test   # test 정답  

        # dataset 설정값
        self.itemnum = itemnum                     # 전체 아이템 개수
        self.maxlen = maxlen                       # sequence 최대 길이
        self.mask_prob = mask_prob                 # trian 시 Mask 확률
        self.neg_sample_size = neg_sample_size     # negative sampling 개수
        self.mask_token = itemnum + 1              # BERT-style MASK Token
        self.mode = mode                           # 현재 dataet 모드
        self.item_size = itemnum                   # DataModule에서 사용되는 item 크기

        # 사용할 유저 목록 생성
        if mode == 'train':
            # train에서는 sequence 길이가 2 이상인 유저만 사용
            # (최소 1개는 context, 1개는 예측 대상이 필요)
            self.users = [
                u for u, seq in user_train.items()
                if len(seq) >= 2
            ]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))
        else:
            # valid / test 에서는 정답이 있는 유저만 사용
            ref = user_valid if mode == 'valid' else user_test

            self.users = [
                u for u in user_train
                if ref.get(u)
            ]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))


    def __len__(self):
        """
        Dataset의 전체 데이터 개수 반환
        DataLoader가 iteration 길이를 계산할 때 사용
        """
        return len(self.users)
    
    def __getitem__(self, idx):
        """
        DataLoader가 batch를 만들 때 호출하는 함수
        idx에 해당하는 user의 데이터를 반환
        """

        # 현재 index에 해당하는 user ID
        u = self.users[idx]

        if self.mode == 'train':
            # train 모드 -> 랜덤 masking 적용
            return self._train_item(u)
        else: 
            # valid / test 모드 -> 마지막 아이템 예측
            return self._eval_item(u)

    def _train_item(self, u):
        """
        train mode에서 호출되는 함수
        BERT4REC 방식의 random masking을 적용하여 모델이 가려진 아이템을 맞추도록 학습 데이터를 생성
        """

        # user u의 sequence 추출
        seq = self.user_train[u]

        tokens, labels = [], []

        # sequence의 각 아이템에 대해 masking 여부 결정
        for item in seq:

            # mask_prob 확률로 아이템을 MASK 토큰으로 치환
            if random.random() < self.mask_prob:

                tokens.append(self.mask_token) # 입력에는 MASK 넣고
                labels.append(item)            # 정답은 원래 item 저장
            else:
                tokens.append(item)            # 그대로 입력
                labels.append(0)               # 학습에서 무시 (ignore)

        # sequence 길이가 maxlen보다 길면 뒤에서부터 자름
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]

        # 부족한 길이는 padding
        pad_len = self.maxlen - len(tokens)

        # 앞쪽에 PAD(0) 추가
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _eval_item(self, u):
        """
        validation / test 에서 사용하는 데이터 생성
        마지막 아이템을 맞추는 Next Item Prediction 방식
        """

        # train sequence 가져오기
        train_seq = self.user_train.get(u, [])

        # validation / test 분기
        if self.mode == 'valid':
            # validation은 train history 기반
            history = train_seq
            # validation 정답 아이템
            target = self.user_valid[u][0]
        else:
            # test는 train + valid history 사용
            history = self.user_train[u] + self.user_valid.get(u, [])
            # test 정답 아이템
            target = self.user_test[u][0]
        
        # 입력 sequence 생성

        # history에서 item만 추출
        item_seq = [iid for iid in history]

        # maxlen - 1 만큼만 사용
        item_seq = item_seq[-(self.maxlen - 1):]

        # 마지막 위치를 MASK로 설정
        seq = item_seq + [self.mask_token]

        # padding
        pad_len = self.maxlen - len(seq)
        seq = [0] * pad_len + seq

        # Negative sampling

        # 이미 본 아이템은 negative로 뽑지 않기 위해 집합 생성
        rated = set(self.user_train[u]); rated.add(0)
        negs = []

        # neg_sample_size 만큼 negative 아이템 생성
        while len(negs) < self.neg_sample_size:
            # 랜덤 아이템 선택
            t = np.random.randint(1, self.itemnum + 1)
            # 이미 본 아이템이 아니면 추가
            if t not in rated:
                negs.append(t)
            
        # candidate item 리스트 (첫 번째가 정답)
        candidates = [target] + negs # 총 101개 (1 positive + 100 negative)

        # label 생성 (target = 1 / negative = 0)
        labels = [1] + [0] * self.neg_sample_size

        # tensor 형태로 반환
        return (
            torch.LongTensor(seq),        # 입력 sequence (maxlen)
            torch.LongTensor(candidates), # candidate item (101)
            torch.LongTensor(labels)      # 정답 여부
        )
        
