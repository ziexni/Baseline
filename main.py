import torch
import pandas as pd
import numpy as np
from mmmlp import *
from utils import *
import json
import time
import argparse
import os
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--intr_dataset', default='datasets//interaction.parquet')
parser.add_argument('--dataset', default='datasets//interaction.parquet')
parser.add_argument('--train_dir', default='train_output', help='학습 데이터 저장 디렉토리')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0001, type=float) # 0.001
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=128, type=int) # 100
parser.add_argument('--num_blocks', default=3, type=int) # 2
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu') # GPU 사용
# parser.add_argument('--inference_only', default=False, type=str2bool) 
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False) # Post-LN

args = parser.parse_args()

if __name__ == '__main__':

    # Data load
    dataset = data_partition(args.intr_dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    print("users: ", usernum)
    print("items: ", itemnum)

    # multimodal feature load
    title_dataset = np.load('datasets/title_emb.npy')
    item_dataset = pd.read_parquet('datasets/item_used.parquet')

    with open('datasets/category2id.json', 'r') as f:
        category2id = json.load(f)

    # 모델 생성
    model = MMMLP(
        args,
        intr_dataset = pd.read_parquet(args.intr_dataset),
        title_dataset = title_dataset,
        item_dataset = item_dataset,
        category2id = category2id
    ).to(args.device)

    print("Model initialized")

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4 # L2 정규화
    )

    print('Building item frequency ... ')
    item_freq = build_item_freq(user_train, itemnum)

    # sampler
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
        item_freq=item_freq
    )

    # training
    num_batch = len(user_train) // args.batch_size + 1

    folder = args.train_dir
    os.makedirs(folder, exist_ok=True)

    # 결과 기록 파일
    f = open(os.path.join(folder, 'log.txt'), 'w')
    
    # Initialization
    t0 = time.time()
    T = 0.0
    best_val_ndcg = 0.0
    best_val_hr = 0.0
    best_val_mrr = 0.0
    best_test_ndcg = 0.0
    best_test_hr = 0.0
    best_test_mrr = 0.0

    # Early Stopping
    patience = 20
    wait = 0

    try:
        for epoch in range(1, args.num_epochs + 1):

            model.train()
            epoch_loss = 0

            for step in tqdm(range(num_batch),
                            total=num_batch,
                            ncols=70,
                            leave=False,
                            unit='b'):

                u, seq, pos, neg = sampler.next_batch()

                seq = torch.LongTensor(np.array(seq)).to(args.device)
                pos = torch.LongTensor(np.array(pos)).to(args.device)
                neg = torch.LongTensor(np.array(neg)).to(args.device)

                seq_len = (seq != 0).sum(dim=1)
 
                optimizer.zero_grad()

                seq_output = model(seq, seq_len) # [batch, hidden]

                pos_last = pos[:, -1]
                neg_last = neg[:, -1]

                pos_emb = model.get_item_multimodal(pos_last)
                neg_emb = model.get_item_multimodal(neg_last)

                pos_score = torch.sum(seq_output * pos_emb, dim=-1)
                neg_score = torch.sum(seq_output * neg_emb, dim=-1)

                loss = -F.logsigmoid(pos_score - neg_score).mean()

                loss.backward()
                # Gradient clipping: gradient exploding 방지
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= num_batch

            print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")

            # 20 epoch마다 평가
            if epoch % 20 == 0:

                print("Evaluating...")
                model.eval()

                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)

                print('epoch:%d, valid (NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f), test (NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f)'
                    % (epoch, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1], t_test[2]))

                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()

                # Early Stopping 실제 작동
                if t_valid[0] > best_val_ndcg:
                    best_val_ndcg  = t_valid[0]
                    best_val_hr    = t_valid[1]
                    best_val_mrr   = t_valid[2]
                    best_test_ndcg = t_test[0]
                    best_test_hr   = t_test[1]
                    best_test_mrr  = t_test[2]
                    wait = 0
                    # 최적 모델 save
                    torch.save(model.state_dict(),
                               os.path.join(folder, 'best_model.pt'))
                    print(f'Bast model saved (val: NDCG: {best_val_ndcg})')
                else:
                    wait += 1
                    print(f'No improvement ({wait}/{patience})')
                    if wait >= patience:
                        print('Early stopping triggered')
                        break

    except Exception as e:
        print("Error:", e)
        sampler.close()
        f.close()
        exit(1)

    sampler.close()
    f.close()

    print(f"\n=== Best Results ===")
    print(f"Valid  NDCG@10: {best_val_ndcg:.4f}  HR@10: {best_val_hr:.4f}  MRR: {best_val_mrr:.4f}")
    print(f"Test   NDCG@10: {best_test_ndcg:.4f}  HR@10: {best_test_hr:.4f}  MRR: {best_test_mrr:.4f}")
    print("Done")
