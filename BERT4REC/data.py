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
    'train'       : random masking → (seq, pos_labels, dummy_neg)
    'valid'/'test': 마지막 위치 MASK → (seq, candidates, labels)
                    candidates[0] = 정답, candidates[1:] = negatives
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
            # 학습: 시퀀스 길이 2 이상인 전체 유저 (제한 없음)
            self.users = [
                u for u, seq in user_train.items()
                if len(seq) >= 2
            ]
        else:
            ref       = user_valid if mode == 'valid' else user_test
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
        seq    = self.user_train[u]
        tokens = []
        labels = []

        for item in seq:
            if random.random() < self.mask_prob:
                # ✅ 원본 레포와 동일: MASK 토큰만 사용
                tokens.append(self.mask_token)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)   # 0 = ignore (CE loss ignore_index=0)

        # maxlen으로 자르기 (최근 시퀀스 유지)
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]

        # 앞을 0으로 padding
        pad_len = self.maxlen - len(tokens)
        tokens  = [0] * pad_len + tokens
        labels  = [0] * pad_len + labels

        # neg 자리는 dummy 0 (CE loss라 사용 안 함, dataloader 구조 유지)
        return (
            torch.LongTensor(tokens),
            torch.LongTensor(labels),
            torch.LongTensor([0] * self.maxlen)
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
