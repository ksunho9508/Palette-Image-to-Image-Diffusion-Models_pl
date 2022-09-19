from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    QuantizationAwareTraining,
    ModelPruning,
)

import os


def get_callbacks(conf):
    print("CALLBACK SETTING")
    cb_list = []

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(conf["trial_save_dir"], "checkpoints")
        if conf["optuna"]
        else os.path.join(conf["save_dir"], "checkpoints"),
        auto_insert_metric_name=False,
        filename="{epoch}-auroc{hp/val_auroc:.3f}-auprc{hp/val_auprc:.3f}-fpr{hp/val_fpr@tpr95:.3f}",
        monitor="hp/val_total_criteria",
        mode="max",
        save_top_k=-1,
    )
    cb_list.append(ckpt_callback)
    if "callbacks" not in conf:
        return cb_list

    if "early_stopping" in conf["callbacks"]:
        print("\tEarly stopping")
        earlystop_callback = EarlyStopping(
            monitor="val/loss_total",
            patience=int(conf["max_epochs"] / 5) + 1,
            mode="min",
        )
        cb_list.append(earlystop_callback)

    if "QuantizationAwareTraining" in conf["callbacks"]:
        print("\tQuantizationAwareTraining")
        qat_callbacks = QuantizationAwareTraining(input_compatible=True)
        cb_list.append(qat_callbacks)

    if "ModelPruning" in conf["callbacks"]:
        print("\ModelPruning")
        pruning_callbacks = ModelPruning(
            "l1_unstructured", amount=compute_pruning_amount
        )
        cb_list.append(pruning_callbacks)

    return cb_list


def compute_pruning_amount(epoch):
    # the sum of all returned values need to be smaller than 1
    if epoch == 0:
        return 0.9

    elif epoch == 3:
        return 0.6

    elif epoch > 6:
        return 0.3
