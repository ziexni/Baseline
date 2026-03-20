import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
from collections import defaultdict
import random
import sys
import copy


# ── Negative Sampling ────────────────────────────────────────────────────────
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def popularity_neq(item_ids, probs, s):
    while True:
        t = np.random.choice(item_ids, p=probs)
        if t not in s:
            return int(t)

def build_item_freq(user_train, itemnum):
    freq = np.zeros(itemnum + 2, dtype=np.float32)
    for items in user_train.values():
        for item in items:
            freq[item] += 1
    item_ids = np.arange(1, itemnum + 1, dtype=np.int32)
    counts   = np.power(freq[item_ids], 0.75)
    probs    = counts / counts.sum()
    return item_ids, probs


# ── Batch Sampling (Worker) ──────────────────────────────────────────────────
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, item_freq=None):
    def sample(uid):
        uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if item_freq is not None:
                neg[idx] = popularity_neq(item_freq[0], item_freq[1], ts)
            else:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt  = i
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids    = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


# ── WarpSampler ──────────────────────────────────────────────────────────────
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, item_freq=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors   = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(User, usernum, itemnum, batch_size, maxlen,
                          self.result_queue, np.random.randint(2e9), item_freq)
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# ── Data Partition ───────────────────────────────────────────────────────────
def data_partition(fname):
    df = pd.read_parquet(fname)
    # [FIX] SASRec과 동일하게 정렬 기준 맞춤
    df = df.sort_values(by=['user_id', 'timestamp', 'item_id'], kind='mergesort')

    df['item_id'] = df['item_id'] + 1
    df['user_id'] = df['user_id'] + 1

    usernum = df['user_id'].max()
    itemnum = df['item_id'].max()

    User       = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test  = {}

    for u, i in zip(df['user_id'], df['item_id']):
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        # [FIX] SASRec과 동일: nfeedback < 3 기준
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user]  = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user]  = [User[user][-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]


# ── Evaluate on Test Set (SASRec 방식과 동일하게 맞춤) ───────────────────────
def evaluate(model, dataset, args, batch_size=256):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # [FIX] SASRec과 동일: 500명 샘플링
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 500)
    else:
        users = range(1, usernum + 1)

    valid_users = [u for u in users if len(train[u]) >= 1 and len(test[u]) >= 1]

    NDCG, HT, MRR, valid_user = 0.0, 0.0, 0.0, 0.0

    for batch_start in range(0, len(valid_users), batch_size):
        batch_users = valid_users[batch_start:batch_start + batch_size]

        seqs, item_idxs = [], []
        for u in batch_users:
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1

            # [FIX] SASRec과 동일: valid 아이템을 시퀀스 끝에 포함
            seq[idx] = valid[u][0]
            idx -= 1

            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            candidates = [test[u][0]]
            # [FIX] SASRec과 동일: 100개 negative (총 101개)
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                candidates.append(t)

            seqs.append(seq)
            item_idxs.append(candidates)

        seqs      = np.array(seqs)
        item_idxs = np.array(item_idxs)

        preds = -model.predict_batch(seqs, item_idxs)

        for pred in preds:
            rank = pred.argsort().argsort()[0].item()
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT   += 1
            MRR += 1 / (rank + 1)

    return NDCG / valid_user, HT / valid_user,  MRR / valid_user


# ── Evaluate on Valid Set (SASRec 방식과 동일하게 맞춤) ─────────────────────
def evaluate_valid(model, dataset, args, batch_size=256):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # [FIX] SASRec과 동일: 500명 샘플링
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    valid_users = [u for u in users if len(train[u]) >= 1 and len(valid[u]) >= 1]

    NDCG, HT, MRR, valid_user = 0.0, 0.0, 0.0, 0.0

    for batch_start in range(0, len(valid_users), batch_size):
        batch_users = valid_users[batch_start:batch_start + batch_size]

        seqs, item_idxs = [], []
        for u in batch_users:
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1

            # [FIX] SASRec valid: train만으로 시퀀스 구성 (valid 미포함)
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            candidates = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                candidates.append(t)

            seqs.append(seq)
            item_idxs.append(candidates)

        seqs      = np.array(seqs)
        item_idxs = np.array(item_idxs)

        preds = -model.predict_batch(seqs, item_idxs)

        for pred in preds:
            rank = pred.argsort().argsort()[0].item()
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT   += 1
            MRR += 1 / (rank + 1)

    return NDCG / valid_user, HT / valid_user, MRR / valid_user