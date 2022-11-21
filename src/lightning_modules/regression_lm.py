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
 
class DE_RegressionLM(LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf) 
        self.model = baseUNet(in_channel=conf['in_channel'], out_channel=conf['out_channel']) 

        self.de_name = ['gaussian', 'spot', 'halo', 'hole']     
        self.debug = True if conf['debug'] else False
        self.loss = torch.nn.MSELoss()
        self.total_loss = {'train':0, 'val':0, 'test':0} 
        self.metrics = {'fid': FID().to(self.device), 
                        'is': IS().to(self.device), 
                        'psnr': PSNR().to(self.device), 
                        'ssim': SSIM().to(self.device)
                        } 

    def step(self, batch, batch_idx=None, tvt="train"):
        y_0 = batch['gt_image']
        y_cond = batch['cond_image'] 

        if tvt == 'val' and self.debug:
            save_image(batch['gt_image'], 'gt_image.jpg', normalize=True) 
            save_image(batch['cond_image']*0.5 + 0.5, 'cond_image.jpg')  
 
        y_out = self.model(y_cond)
        if tvt == 'train':
            loss = self.loss(y_out, y_0)
            return loss
        else:
            if batch_idx == 0:
                self.visualize_sr_out(y_0, y_cond, y_out)  
            self.update_metrics(pred=y_out.detach().float(), target=y_0.detach()),  
        
     
    def visualize_sr_out(self, y_0, y_cond, y_out):
        save_dir = os.path.join(self.hparams.save_dir, str(self.current_epoch))
        os.makedirs(save_dir, exist_ok=True) 

        res = torch.cat([y_0, y_cond, y_out], dim=0)
        save_image(res, save_dir + f'/{self.local_rank}_sr_out.jpg', nrow = y_0.size(0), normalize=True) 

    def update_metrics(self, pred, target):
        for m in ['psnr', 'ssim']:
            self.metrics[m].update(pred, target)       

        pred_uint8 = ((pred * 0.5 + 0.5) * 255).type(torch.uint8) 
        target_uint8 = ((target * 0.5 + 0.5) * 255).type(torch.uint8) 
        for m in ['is']:
            self.metrics[m].update(pred_uint8)        
        for m in ['fid']:
            self.metrics[m].update(target_uint8, real=True)
            self.metrics[m].update(pred_uint8, real=False)

    def compute_metrics(self, tvt='val'):
        tot = []
        for n, m in self.metrics.items():
            res = m.compute()
            if n == 'is':
                res = res[0]
            print(n, m, res)
            self.log(f"{tvt}/{n}", res, rank_zero_only=True)
            tot.append(res.item() if n != 'fid' else (1/res).item())
            m.reset() 
        comprehensive_value = gmean(tot)
        self.log(f"{tvt}/comprehensive_value", comprehensive_value, rank_zero_only=True) 

  
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, tvt="train") 
        if self.global_rank == 0 and batch_idx % self.hparams.sample_iter == 0:
            self.logging_loss(loss, tvt='train', on_step=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"])

        return {"loss": loss}

    def on_validation_start(self):
        for m in self.metrics.values():
            m = m.to(self.device)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx=batch_idx, tvt="val")    

    def on_validation_epoch_end(self) -> None:  
        self.compute_metrics()

    def logging_loss(self, x, tvt, on_epoch=False, on_step=False, sync_dist=False):   
        self.total_loss[tvt] = self.total_loss[tvt] * 0.9 + x * 0.1         
        self.log(
            f'{tvt}/loss',
            self.total_loss[tvt],  
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=sync_dist
        ) 
     
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
 