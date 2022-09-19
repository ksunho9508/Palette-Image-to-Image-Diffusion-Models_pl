from locale import normalize
import os
import torch
from torch import optim
from torchmetrics import AUROC
from torch.nn import functional as F
from torchvision.utils import save_image
from pytorch_lightning import LightningModule
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from scipy.stats import gmean
from .models.diffusion_model import DiffusionModel
import pandas as pd  
 
class DiffusionLM(LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        self.model = DiffusionModel(conf, device = self.device)  
        if 'pretrain_weight' in conf:
            task = conf['task']
            self.load_from_pretrain(os.path.join(conf['pretrain_weight'], f'{task}.pth'))
        
        self.total_loss = 0
        self.cnt = 0

    def load_from_pretrain(self, pretrain_dir):
        self.model.set_new_noise_schedule(phase='train', device=self.device)
        self.model.load_state_dict(torch.load(pretrain_dir, map_location=self.device), strict=True)
        print('loaded pretrained weight')

    def step(self, batch, batch_idx=None, tvt="train"):
        y_0 = batch['gt_image']
        y_cond = batch['cond_image']
        m = batch['mask']
        if tvt == 'val':
            save_image(batch['gt_image'], 'gt_image.jpg', normalize=True) 
            save_image(batch['mask_image'], 'mask_image.jpg', normalize=True) 
        
        if tvt == 'train':
            loss = self.model(y_0, y_cond, mask=m, device=self.device)
            return  loss
        else:
            y_intermediate = self.model.restoration(
                y_cond=y_cond, 
                y_t=y_cond,
                y_0=y_0,
                mask=m,
                sample_num=8, 
                device=self.device)
            self.visualize_restoration(y_0, y_cond, y_intermediate)            
        
    def on_train_start(self) -> None:
        self.model.set_new_noise_schedule(phase='train', device=self.device)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, tvt="train") 
        if self.global_rank == 0 and batch_idx % self.hparams.sample_iter == 0:
            self.logging_smth(loss, log_name='loss', on_step=True)
            self.logging_smth(self.trainer.optimizers[0].param_groups[0]["lr"], 'lr', on_step=True)
        self.cnt += 1
        return {"loss": loss}

    def training_epoch_end(self, outputs): 
        self.cnt = 0 

    def on_validation_start(self):
        self.model.set_new_noise_schedule(phase='val', device=self.device) 

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx=batch_idx, tvt="val")  
    
    def logging_smth(self, x, log_name, on_epoch=False, on_step=False):
        if log_name == 'loss':
            self.total_loss += x
        self.log(
            log_name,
            x,  
            on_epoch=on_epoch,
            on_step=on_step,
        ) 
    
    def visualize_restoration(self, y_0, y_cond, y_intermediate=None):
        save_dir = os.path.join(self.hparams.save_dir, str(self.current_epoch))
        os.makedirs(save_dir, exist_ok=True)
        save_image(y_0, save_dir + f'/{self.local_rank}_y_0.jpg', normalize=True)
        save_image((y_cond * 0.5) + 0.5, save_dir + f'/{self.local_rank}_y_cond.jpg') 
        save_image((y_intermediate * 0.5) + 0.5, save_dir + f'/{self.local_rank}_process.jpg', nrow= y_0.size(0))
        save_image(y_intermediate[-1 * y_0.size(0):], save_dir + f'/{self.local_rank}_y_hat.jpg', normalize=True)


    def configure_optimizers(self):
        optim_name = self.hparams.optimizer
        if optim_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.max_lr)
        elif optim_name == "adamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.max_lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif optim_name == "radam":
            optimizer = optim.RAdam(self.model.parameters(), lr=self.hparams.max_lr)
        else:
            raise NotImplementedError(f"{optim_name} is not implemented.")

        iter_per_epoch = self.trainer.datamodule.get_iter_per_epoch()
        total_iter = iter_per_epoch * self.hparams.max_epochs
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_iter // self.hparams.lr_scheduler_cycles,
            cycle_mult=1.0,
            max_lr=self.hparams.max_lr,
            min_lr=self.hparams.min_lr,
            warmup_steps=int(
                (total_iter // self.hparams.lr_scheduler_cycles)
                * self.hparams.lr_scheduler_warmup_rate
            ),
            gamma=self.hparams.lr_scheduler_gamma,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
 