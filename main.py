import os
import sys
from scipy.stats import gmean, hmean

from parser import parse_argument
from callbacks import get_callbacks
from src.data_modules import get_data_module
from src.lightning_modules import get_lightning_module

import pytorch_lightning as pl
import torch
import torch.multiprocessing
from collections import defaultdict


def run(conf):
    if conf["only_test"]:
        conf["gpu"] = [3]
        conf["batch_size"] = 1
        dm = get_data_module(conf)
        lm = get_lightning_module(conf)
        trainer = pl.Trainer(
            default_root_dir=conf["save_dir"],
            devices=conf["devices"] if not conf["gpu"] else conf["gpu"],
            accelerator="gpu",
            logger=pl.loggers.TensorBoardLogger(
                save_dir=conf["save_dir"],
                default_hp_metric=False,
            ),
        )
        trainer.test(lm, datamodule=dm)
    else:
        dm = get_data_module(conf)
        lm = get_lightning_module(conf)
        cb = get_callbacks(conf)

        trainer = pl.Trainer(
            logger=pl.loggers.TensorBoardLogger(
                save_dir=conf["save_dir"],
                default_hp_metric=False,
            ),
            callbacks=cb,
            default_root_dir=conf["save_dir"],
            devices=conf["devices"],
            max_epochs=conf["max_epochs"],
            accelerator="gpu",
            strategy="ddp_spawn_find_unused_parameters_false"
            if conf["devices"] > 1
            else None,
            precision=16,
            num_sanity_val_steps=1,
            check_val_every_n_epoch=5,
            limit_val_batches=1,
            sync_batchnorm=True,
        )
        trainer.fit(lm, datamodule=dm)


if __name__ == "__main__":
    global conf
    conf = parse_argument()
    conf_ = defaultdict(lambda: None)
    for k, v in conf.items():
        conf_[k] = v
    conf = conf_
    os.makedirs(conf["save_dir"], exist_ok=True)
    run(conf)
