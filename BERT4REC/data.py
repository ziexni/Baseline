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

        else:
            ref       = user_valid if mode == 'valid' else user_test
            all_users = [u for u in user_train if ref.get(u)]
            self.users = all_users
            # ✅ BERT4Rec 원본: 유저 수 제한 없음

        # ✅ BERT4Rec 원본: neg을 사전에 고정 생성 (매번 재샘플링 X)
        if mode in ('valid', 'test'):
            self._precompute_negatives()

    def _precompute_negatives(self):
        """
        원본 BERT4Rec: negative를 사전에 고정 생성
        - valid: train items 제외
        - test : train + valid 정답 제외  ← 핵심
        """
        self.user_negatives = {}
        for u in self.users:
            # ✅ BERT4Rec 원본: test 시 valid 정답도 rated에 포함
            rated = set(self.user_train[u])
            rated.add(0)
            if self.mode == 'test':
                rated.update(self.user_valid.get(u, []))  # valid 정답 제외

            negs = []
            while len(negs) < self.neg_sample_size:
                t = np.random.randint(1, self.itemnum + 1)
                if t not in rated:
                    negs.append(t)
            self.user_negatives[u] = negs

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        if self.mode == 'train':
            return self._train_item(u)
        else:
            return self._eval_item(u)

    def _train_item(self, u):
        seq    = self.user_train[u]
        tokens = []
        labels = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:
                # ✅ 원본 80/10/10
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(random.randint(1, self.itemnum))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]

        pad_len = self.maxlen - len(tokens)
        tokens  = [0] * pad_len + tokens
        labels  = [0] * pad_len + labels

        # ✅ 원본과 동일: 2개 반환
        return (
            torch.LongTensor(tokens),
            torch.LongTensor(labels)
        )

    def _eval_item(self, u):
        train_seq = self.user_train.get(u, [])

        if self.mode == 'valid':
            history = train_seq
            target  = self.user_valid[u][0]
        else:
            # test: train + valid 포함
            history = train_seq + self.user_valid.get(u, [])
            target  = self.user_test[u][0]

        item_seq = history[-(self.maxlen - 1):]
        seq      = item_seq + [self.mask_token]
        pad_len  = self.maxlen - len(seq)
        seq      = [0] * pad_len + seq

        # ✅ 사전 고정된 negatives 사용
        negs       = self.user_negatives[u]
        candidates = [target] + negs
        labels     = [1] + [0] * self.neg_sample_size

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels)
        )



    def _eval_item(self, u):
        train_seq = self.user_train.get(u, [])

        if self.mode == 'valid':
            history = train_seq
            target  = self.user_valid[u][0]
        else:
            # test: train + valid 시퀀스 포함 (SASRec/GRU4Rec 동일)
            history = train_seq + self.user_valid.get(u, [])
            target  = self.user_test[u][0]

        # 마지막 위치에 MASK 추가 (BERT4Rec eval 방식)
        item_seq = history[-(self.maxlen - 1):]
        seq      = item_seq + [self.mask_token]
        pad_len  = self.maxlen - len(seq)
        seq      = [0] * pad_len + seq

        # ✅ SASRec 맞춤: rated = train only + {0}
        #    test여도 valid 정답을 rated에 추가하지 않음
        rated = set(self.user_train[u])
        rated.add(0)

        negs = []
        while len(negs) < self.neg_sample_size:
            t = np.random.randint(1, self.itemnum + 1)
            if t not in rated:
                negs.append(t)

        # candidates[0] = 정답 (ordering 보장)
        candidates = [target] + negs
        labels     = [1] + [0] * self.neg_sample_size

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels)
        )
