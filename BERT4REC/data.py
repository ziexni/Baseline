"""
data.py
MovieLens -> MicroVideo 데이터셋으로 교체
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle

def get_data(interaction_path):
    """
    Interaction file load 
    Returns :
        user_train, user_valid, user_test : [(item_id)]
        usernum, itemnum
    """
    with open(interaction_path, 'rb') as f:
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
        if n < 3:
            user_train[user] = seq
            user_valid[user] = []
            user_test[user] = []
        else:
            # ✅ Leave-two-out split
            user_train[user] = seq[:-2]   # 마지막 2개 제외
            user_valid[user] = [seq[-2]]  # 뒤에서 두 번째
            user_test[user]  = [seq[-1]]  # 마지막

    print(f"[Split] train users: {len(user_train)}, "
          f"valid: {sum(1 for v in user_valid.values() if v)}, "
          f"test: {sum(1 for v in user_test.values() if v)}")

    return user_train, user_valid, user_test, usernum, itemnum

class MicroVideoDataset(Dataset):
    """
    BERT4REC Cloze task Dataset

    [ 토큰 규칙 ]
    PAD = 0
    item = 1 ~ itemnum
    MASK = itemnum + 1

    [ mode ]
    'train'       : random masking → (seq, labels)
    'valid'/'test': 마지막 위치 MASK → (seq, candidates, labels)
                    candidates = [정답] + 100 negatives  (index 0이 항상 정답)
    """
    def __init__(self, user_train, user_valid, user_test,
                 itemnum, maxlen, mask_prob=0.2, neg_sample_size=100, mode='train', usernum=0):
        
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test  = user_test

        self.itemnum        = itemnum
        self.maxlen         = maxlen
        self.mask_prob      = mask_prob
        self.neg_sample_size = neg_sample_size
        self.mask_token     = itemnum + 1
        self.mode           = mode
        self.item_size      = itemnum

        if mode == 'train':
            self.users = [
                u for u, seq in user_train.items()
                if len(seq) >= 2
            ]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))
        else:
            ref = user_valid if mode == 'valid' else user_test
            self.users = [
                u for u in user_train
                if ref.get(u)
            ]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))


    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        u = self.users[idx]
        if self.mode == 'train':
            return self._train_item(u)
        else: 
            return self._eval_item(u)

    def _train_item(self, u):
        seq = self.user_train[u]
        tokens, labels = [], []

        for item in seq:
            if random.random() < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)

        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]

        pad_len = self.maxlen - len(tokens)
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _eval_item(self, u):
        train_seq = self.user_train.get(u, [])

        if self.mode == 'valid':
            history = train_seq
            target  = self.user_valid[u][0]
        else:
            history = train_seq + self.user_valid.get(u, [])
            target  = self.user_test[u][0]
        
        item_seq = [iid for iid in history]
        item_seq = item_seq[-(self.maxlen - 1):]
        seq      = item_seq + [self.mask_token]

        pad_len = self.maxlen - len(seq)
        seq = [0] * pad_len + seq

        # ✅ Fix: rated에 train + valid 정답 모두 포함해서 negative 오염 방지
        rated = set(self.user_train[u])
        rated.add(0)
        if self.mode == 'test':
            # valid 정답(seq[-2])이 test negative로 뽑히지 않도록 제외
            rated.update(self.user_valid.get(u, []))

        negs = []
        while len(negs) < self.neg_sample_size:
            t = np.random.randint(1, self.itemnum + 1)
            if t not in rated:
                negs.append(t)
            
        # ✅ candidates[0] = 정답 (ordering 보장)
        candidates = [target] + negs          # 101개 (1 pos + 100 neg)
        labels     = [1] + [0] * self.neg_sample_size

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels)
        )
