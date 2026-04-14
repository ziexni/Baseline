import argparse # CLI argument 파싱
import datetime # 로그 폴더 timestamp 생성용

# 내부 모듈
from datamodule import DataModule # 데이터 로딩 + DataLoader 관리
from lit_model import BERT4REC 

# Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger # 로그 시각화
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

def _setup_parser():
    """
    CLI argument parser 구성
    """
    parser = argparse.ArgumentParser(description='BERT4REC - MicroVideo')

    # Data 관련 argument
    data_group = parser.add_argument_group("Data Args")
    DataModule.add_to_argparse(data_group)

    # Model 관련 argument
    model_group = parser.add_argument_group("Model Args")
    BERT4REC.add_to_argparse(model_group)

    return parser

def _set_trainer_args(args):
    """
    코드에서 강제로 Trainer 설정 덮어쓰기
    (CLI보다 우선 적용됨)
    """

    # 기본 학습 설정
    args.max_epochs = 100                 # 최대 epoch 
    args.gradient_clip_val = 5.0          # gradient clipping (폭주 방지)
    args.gradient_clip_algorithm = "norm" # L2 Norm 기반 clipping

     # 로그 저장 경로 
    args.save_dir = "Training/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = "BERT4REC_MicroVideo"

    # Early Stopping 기준 
    args.monitor = "val_loss"    # 이 값 기준으로 best 판단
    args.mode = "min"            # 낮을수록 좋음
    args.patience = 5            # 5 epoch 동안 개선 없으면 종료

    # logging 설정 
    args.logging_interval = "step"  # step 단위로 로그 기록

def main():
    """
    전체 학습 파이프라인 실행
    """

    # argument parsing
    parser = _setup_parser()
    args = parser.parse_args()

    # 코드 내부 설정 적용
    _set_trainer_args(args)
            
    # ===== DataModule 생성 =====
    # 내부에서:
    # - interaction 로딩
    # - train/valid/test split
    # - args.item_size 자동 설정
    data = DataModule(args)

    # 모델 생성
    lit_model = BERT4REC(args)

    # Logger
    logger = TensorBoardLogger(
        save_dir = args.save_dir,
        name = args.name
    )

    # EarlyStopping
    early_stop = EarlyStopping(
        monitor = args.monitor,
        mode = args.mode,
        patience = args.patience
    )

    # Learning Rate 추적 
    lr_monitor = LearningRateMonitor(
        logging_interval = args.logging_interval
    )

    checkpoint = ModelCheckpoint(
    monitor    = 'val_loss',
    mode       = 'min',
    save_top_k = 1,
    filename   = 'best'
    )

    # 변경
    trainer = Trainer(
        max_epochs             = args.max_epochs,
        gradient_clip_val      = args.gradient_clip_val,
        gradient_clip_algorithm= args.gradient_clip_algorithm,
        check_val_every_n_epoch = 5,
        logger                 = logger,
        callbacks              = [early_stop, lr_monitor, checkpoint],
        accelerator            = 'gpu',
        devices                = 1,
    )

    # 학습 
    trainer.fit(lit_model, datamodule=data)

    # 테스트 
    trainer.test(lit_model, datamodule=data, ckpt_path='best')


if __name__ == "__main__":
    main()

