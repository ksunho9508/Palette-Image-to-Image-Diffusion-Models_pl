from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .augments.aug_albumentation import aug_train, aug_val, A
from .datasets.inhouse_dataset import (
    Fundus_InpaintDataset,
    Fundus_UncropDataset,
    Fundus_EnhancementDataset,
    Fundus_DownstreamDataset,
    Fundus_Finding_InpaintDataset,
    Fundus_BaseDataset
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
        self.task = conf['task']
        if conf["task"] == "inpainting":
            self.dataset = Fundus_InpaintDataset
        elif conf["task"] == "uncropping":
            self.dataset = Fundus_UncropDataset
        elif conf['task'] in ['vessel', 'optic_fovea', 'finding']:
            self.dataset = Fundus_DownstreamDataset
        elif conf['task'] in ['finding_inpainting', 'finding_inpainting_inference']:
            self.dataset = Fundus_Finding_InpaintDataset
        elif conf['task'] == 'base':
            self.dataset = Fundus_BaseDataset
        else:  # if conf["task"] == "enhancement":
            self.dataset = Fundus_EnhancementDataset
        
        self.mask_mode = conf["mask_mode"]
        self.data_len = 128 if conf["debug"] else -1
        if 'num_of_inference' in conf:
            self.data_len = conf['num_of_inference']
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
                task = self.task
            )

            self.val_dataset = self.dataset(
                self.data_dir,
                split="val",
                image_size=self.image_size,
                mask_mode=self.mask_mode,
                data_len = 8,
                task = self.task
            )
            print(f'train dataset: {len(self.train_dataset)} \n val dataset: {len(self.val_dataset)}')

        if stage == "test":
            self.test_dataset = self.dataset(
                self.data_dir,
                split="train",
                image_size=self.image_size,
                mask_mode=self.mask_mode,
                data_len = self.data_len,
                task = self.task
            )
            print(f'test dataset: {len(self.test_dataset)}')
        
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
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
