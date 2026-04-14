import argparse
import datetime

from datamodule import DataModule
from lit_model  import BERT4REC
from pytorch_lightning import Trainer
from pytorch_lightning.loggers   import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint
)


def _setup_parser():
    parser = argparse.ArgumentParser(description='BERT4REC - MicroVideo')
    data_group  = parser.add_argument_group("Data Args")
    model_group = parser.add_argument_group("Model Args")
    DataModule.add_to_argparse(data_group)
    BERT4REC.add_to_argparse(model_group)
    return parser


def _set_trainer_args(args):
    args.max_epochs = 100

    # ✅ SASRec 맞춤: gradient clip 1.0
    args.gradient_clip_val       = 1.0
    args.gradient_clip_algorithm = "norm"

    args.save_dir = "Training/logs/" + \
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = "BERT4REC_MicroVideo"

    # ✅ SASRec 맞춤: NDCG_val 기준 early stopping
    args.monitor  = "NDCG_val"
    args.es_mode  = "max"        # 높을수록 좋음
    args.patience = 3            # SASRec patience=3

    args.logging_interval = "step"


def main():
    parser = _setup_parser()
    args   = parser.parse_args()
    _set_trainer_args(args)

    data      = DataModule(args)
    lit_model = BERT4REC(args)

    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.name)

    # ✅ SASRec 맞춤: NDCG_val 기준, patience=3
    early_stop = EarlyStopping(
        monitor  = args.monitor,
        mode     = args.es_mode,
        patience = args.patience
    )

    lr_monitor = LearningRateMonitor(logging_interval=args.logging_interval)

    checkpoint = ModelCheckpoint(
        monitor    = args.monitor,
        mode       = args.es_mode,
        save_top_k = 1,
        filename   = 'best'
    )

    trainer = Trainer(
        max_epochs              = args.max_epochs,
        gradient_clip_val       = args.gradient_clip_val,
        gradient_clip_algorithm = args.gradient_clip_algorithm,
        # ✅ SASRec 맞춤: eval every 10 epoch
        check_val_every_n_epoch = 10,
        logger                  = logger,
        callbacks               = [early_stop, lr_monitor, checkpoint],
        accelerator             = 'gpu',
        devices                 = 1,
    )

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data, ckpt_path='best')


if __name__ == "__main__":
    main()
