import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle


def get_data(interaction_path):
    with open(interaction_path, 'rb') as f:
        df = pickle.load(f)

    df['user_id']  = df['user_id']  + 1
    df['video_id'] = df['video_id'] + 1
    df = df.sort_values(by=['user_id', 'timestamp'],
                        kind='mergesort').reset_index(drop=True)

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
            user_test[user]  = []
        else:
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user]  = [seq[-1]]

    print(f"[Split] train users: {len(user_train)}, "
          f"valid: {sum(1 for v in user_valid.values() if v)}, "
          f"test: {sum(1 for v in user_test.values() if v)}")

    return user_train, user_valid, user_test, usernum, itemnum


class MicroVideoDataset(Dataset):
    """
    BERT4Rec Cloze task Dataset

    [ 토큰 규칙 ]
    PAD  = 0
    item = 1 ~ itemnum
    MASK = itemnum + 1

    [ mode ]
    'train'       : random masking → (seq, labels)
    'valid'/'test': 마지막 위치 MASK → (seq, candidates, labels)
                    candidates = [정답] + 100 negatives (index 0이 항상 정답)
    """
    def __init__(self, user_train, user_valid, user_test,
                 itemnum, maxlen, mask_prob=0.2,
                 neg_sample_size=100, mode='train', usernum=0):
        self.user_train      = user_train
        self.user_valid      = user_valid
        self.user_test       = user_test
        self.itemnum         = itemnum
        self.maxlen          = maxlen
        self.mask_prob       = mask_prob
        self.neg_sample_size = neg_sample_size
        self.mask_token      = itemnum + 1
        self.mode            = mode
        self.item_size       = itemnum

        if mode == 'train':
            self.users = [u for u, seq in user_train.items() if len(seq) >= 2]
            # ✅ SASRec 맞춤: train은 전체 유저 사용 (제한 없음)
        else:
            ref = user_valid if mode == 'valid' else user_test
            all_users = [u for u in user_train if ref.get(u)]
            # ✅ SASRec 맞춤: valid 1000명 / test 10000명
            if mode == 'valid' and len(all_users) > 1000:
                self.users = random.sample(all_users, 1000)
            elif mode == 'test' and len(all_users) > 10000:
                self.users = random.sample(all_users, 10000)
            else:
                self.users = all_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        if self.mode == 'train':
            return self._train_item(u)
        else:
            return self._eval_item(u)

    def _train_item(self, u):
        seq   = self.user_train[u]
        rated = set(self.user_train[u])
        tokens, pos_labels, neg_labels = [], [], []
    
        for item in seq:
            prob = random.random()
            if prob < self.mask_prob:
                # ✅ 80/10/10 규칙
                inner = random.random()
                if inner < 0.8:
                    tokens.append(self.mask_token)       # 80%: MASK
                elif inner < 0.9:
                    tokens.append(item)                  # 10%: 원래 아이템 유지
                else:
                    rand_item = np.random.randint(1, self.itemnum + 1)
                    tokens.append(rand_item)             # 10%: 랜덤 교체
    
                pos_labels.append(item)
                neg = np.random.randint(1, self.itemnum + 1)
                while neg in rated:
                    neg = np.random.randint(1, self.itemnum + 1)
                neg_labels.append(neg)
            else:
                tokens.append(item)
                pos_labels.append(0)
                neg_labels.append(0)
    
        tokens     = tokens[-self.maxlen:]
        pos_labels = pos_labels[-self.maxlen:]
        neg_labels = neg_labels[-self.maxlen:]
    
        pad_len    = self.maxlen - len(tokens)
        tokens     = [0] * pad_len + tokens
        pos_labels = [0] * pad_len + pos_labels
        neg_labels = [0] * pad_len + neg_labels
    
        return (
            torch.LongTensor(tokens),
            torch.LongTensor(pos_labels),
            torch.LongTensor(neg_labels)
        )


    def _eval_item(self, u):
        train_seq = self.user_train.get(u, [])

        if self.mode == 'valid':
            history = train_seq
            target  = self.user_valid[u][0]
        else:
            history = train_seq + self.user_valid.get(u, [])
            target  = self.user_test[u][0]

        item_seq = history[-(self.maxlen - 1):]
        seq      = item_seq + [self.mask_token]
        pad_len  = self.maxlen - len(seq)
        seq      = [0] * pad_len + seq

        # ✅ SASRec 맞춤: rated = train + {0} only
        #    test여도 valid 정답을 rated에 추가하지 않음
        rated = set(self.user_train[u])
        rated.add(0)

        negs = []
        while len(negs) < self.neg_sample_size:
            t = np.random.randint(1, self.itemnum + 1)
            if t not in rated:
                negs.append(t)

        candidates = [target] + negs           # candidates[0] = 정답
        labels     = [1] + [0] * self.neg_sample_size

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels)
        )
