import os
from re import M
import torch
from torch import optim  
from torchvision.utils import save_image
from pytorch_lightning import LightningModule
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts 
from .models import baseUNet
from .metrics import * 
from scipy.stats import gmean
 
class DE_DiffLM(LightningModule):
    def __init__(self, conf): 
        super().__init__()
        self.save_hyperparameters(conf) 
        self.model = baseUNet(in_channel=conf['in_channel'], out_channel=conf['out_channel']) 
        self.de_name = ['gaussian', 'spot', 'halo', 'hole']     
        self.debug = True if conf['debug'] else False
        self.loss = torch.nn.MSELoss()
        self.total_loss = {'train':0, 'val':0, 'test':0} 
        self.metrics = {'gaussian': MSE(), 'halo': MSE(), 'hole': MSE(), 'spot': MSE()}  
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        
    def step(self, batch, batch_idx=None, tvt="train"):
        y_0 = batch['gt_image']
        y_cond = batch['cond_image'] 
        de_mask = batch['de_mask'].float()

        if tvt == 'val' and self.debug:
            save_image(batch['gt_image'], 'gt_image.jpg', normalize=True) 
            save_image(batch['cond_image']*0.5 + 0.5, 'cond_image.jpg')  
 
        de_out = self.model(torch.cat([y_0, y_cond], dim=1))
        if tvt == 'train': 
            # loss = self.loss(de_mask, de_out) 
            # blurness
            loss_blurness = self.loss(self.gap(de_mask[:, 0]), self.gap(de_out[:, 0]))
            # spot
            loss_spot = self.loss(de_mask[:, 1], de_out[:, 1]) 
            # halo & hole
            loss_halo_hole = self.loss(de_mask[:, 2:], de_out[:, 2:])  
            return loss_blurness, loss_spot, loss_halo_hole
        else:
            if batch_idx == 0:
                self.visualize_de_out(de_mask, de_out)  
            self.update_metrics(pred=de_out.detach(), target=de_mask.detach()),  
        
     
    def visualize_de_out(self, de_mask, de_out):
        save_dir = os.path.join(self.hparams.save_dir, str(self.current_epoch))
        os.makedirs(save_dir, exist_ok=True)
        de_name = ['gaussian', 'spot', 'halo', 'hole']
        for i in range(4):
            de_m = de_mask[:,i].unsqueeze(1)
            de_o = de_out[:,i].unsqueeze(1)
            res = torch.cat([de_m, de_o], dim=0)
            save_image(res, save_dir + f'/{self.local_rank}_{de_name[i]}.jpg', nrow = de_mask.size(0)) 

    def update_metrics(self, pred, target):
        for i, de in enumerate(self.de_name):
            self.metrics[de].update(pred[:,i], target[:,i])

    def compute_metrics(self, tvt='val'):
        tot = []
        for n, m in self.metrics.items():
            res = m.compute()
            self.log(f"{tvt}/{n}", res, rank_zero_only=True)
            tot.append(res.item())
            m.reset()
        comprehensive_value = gmean(tot)
        self.log(f"{tvt}/comprehensive_value", comprehensive_value, rank_zero_only=True) 
 

    def training_step(self, batch, batch_idx):
        loss_blurness, loss_spot, loss_halo_hole = self.step(batch, tvt="train") 
        if self.global_rank == 0 and batch_idx % self.hparams.sample_iter == 0:
            self.logging_loss(loss_blurness, loss_name= 'loss_blurness', tvt='train', on_step=True)
            self.logging_loss(loss_spot, loss_name= 'loss_spot', tvt='train', on_step=True)
            self.logging_loss(loss_halo_hole, loss_name= 'loss_halo_hole', tvt='train', on_step=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"])

        return {"loss": loss_blurness + loss_spot * 10 + loss_halo_hole}

    def on_validation_start(self):
        for m in self.metrics.values():
            m = m.to(self.device)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx=batch_idx, tvt="val")    

    def on_validation_epoch_end(self) -> None:  
        self.compute_metrics()

    def logging_loss(self, x, loss_name, tvt, on_epoch=False, on_step=False, sync_dist=False):   
        self.total_loss[tvt] = self.total_loss[tvt] * 0.9 + x * 0.1         
        self.log(
            f'{tvt}/{loss_name}',
            self.total_loss[tvt],  
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist
        ) 
    



    def visualize_restoration(self, y_0, y_cond, y_intermediate=None):
        save_dir = os.path.join(self.hparams.save_dir, str(self.current_epoch))
        os.makedirs(save_dir, exist_ok=True)
        save_image(y_0, save_dir + f'/{self.local_rank}_y_0.jpg', normalize=True)
        save_image((y_cond * 0.5) + 0.5, save_dir + f'/{self.local_rank}_y_cond.jpg') 
        save_image((y_intermediate * 0.5) + 0.5, save_dir + f'/{self.local_rank}_process.jpg') #, nrow= y_0.size(0))
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
 