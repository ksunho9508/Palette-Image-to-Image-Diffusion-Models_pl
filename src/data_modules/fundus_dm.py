from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .augments.aug_albumentation import aug_train, aug_val, A
from .datasets.inhouse_dataset import (
    Fundus_InpaintDataset,
    Fundus_UncropDataset,
    Fundus_EnhancementDataset,
)


class FundusDM(LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.pname = conf["pname"]
        self.data_dir = conf["data_dir"]
        self.batch_size = conf["batch_size"]
        self.image_size = conf["image_size"]
        self.num_workers = conf["num_workers"]
        self.devices = conf["devices"]
        if conf["task"] == "inpainting":
            self.dataset = Fundus_InpaintDataset
        elif conf["task"] == "uncropping":
            self.dataset = Fundus_UncropDataset
        else:  # if conf["task"] == "enhancement":
            self.dataset = Fundus_EnhancementDataset
        self.mask_mode = conf["mask_mode"]
        self.data_len = 50 if conf["debug"] else -1

    def get_iter_per_epoch(self):
        return len(self.train_dataset) // (self.batch_size * self.devices)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                self.data_dir,
                split="train",
                image_size=self.image_size,
                mask_mode=self.mask_mode,
                data_len=self.data_len,
            )

            self.val_dataset = self.dataset(
                self.data_dir,
                split="val",
                image_size=self.image_size,
                mask_mode=self.mask_mode,
            )

        if stage == "test":
            self.test_dataset = self.dataset(
                self.data_dir,
                split="test",
                image_size=self.image_size,
                mask_mode=self.mask_mode,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
