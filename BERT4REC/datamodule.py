"""
datamodule.py
MovieLens DataModule -> MicroVIdeo DataModule 
"""

import pytorch_lightning as pl  
from torch.utils.data import DataLoader
from typing import Optional
from data import MicroVideoDataset, get_data

# 기본 interaction 데이터 경로
INTERACTION_PATH = 'kuaishou_preprocess.pkl'

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()

        # 하이퍼파라미터 / dataloader 설정
        self.max_len = args.max_len                   # sequence 최대 길이
        self.mask_prob = args.mask_prob               # masking 확률 (BERT4REC 핵심)
        self.neg_sample_size = args.neg_sample_size   # valid/test 시 negative sampling 개수
        self.pin_memory = args.pin_memory             # GPU 전송 최적화 옵션
        self.num_workers = args.num_workers           # dataloader 병렬 처리 개수
        self.batch_size = args.batch_size             # batch 크기
        self.interaction_path = args.interaction_path # interaction 데이터 경로

        # 데이터 로드
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = get_data(self.interaction_path)
        
        # args에 item_size 주입 => lit_model에서 vocab_size로써 활용됨
        args.item_size = self.itemnum

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.max_len, self.mask_prob,
                mode='train'
                # ✅ usernum 전달 안 함 — BERT4Rec은 유저 수 제한 없음
            )
            self.valid_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.max_len,
                neg_sample_size=self.neg_sample_size,
                mode='valid'
            )
    
        if stage == 'test' or stage is None:
            self.test_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.max_len,
                neg_sample_size=self.neg_sample_size,
                mode='test'
            )

    def train_dataloader(self):
        """
        학습용 dataloader
        - shuffle=True (필수)
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """
        검증용 dataloader
        - shuffle=False (평가라서 순서 유지)
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """
        테스트용 dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    @staticmethod    
    def add_to_argparse(parser):
        """
        CLI에서 사용할 argument 정의
        """
        parser.add_argument('--max_len',         type=int,   default=50)   # sequence 길이
        parser.add_argument('--mask_prob',       type=float, default=0.2)   # masking 비율
        parser.add_argument('--neg_sample_size', type=int,   default=100)   # negative 샘플 수
        parser.add_argument('--batch_size',      type=int,   default=256)   # batch size
        parser.add_argument('--pin_memory',      type=bool,  default=True)  # GPU 최적화
        parser.add_argument('--num_workers',     type=int,   default=4)     # dataloader worker
        parser.add_argument('--item_size',       type=int,   default=0)     # item vocab size (나중에 덮어씀)
        parser.add_argument(
            '--interaction_path',
            default='/kaggle/input/datasets/jieunl2/kuaishou/kuaishou_preprocess.pkl'
        ) # 데이터 경로

        return parser
