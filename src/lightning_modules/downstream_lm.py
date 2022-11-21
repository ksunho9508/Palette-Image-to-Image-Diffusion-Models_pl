import os
from re import M
import torch
from torch import optim  
from torchvision.utils import save_image, draw_keypoints
from pytorch_lightning import LightningModule
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts 
from .models import baseUNet, models
from .metrics import * 
from scipy.stats import gmean
 
class DownstreamLM(LightningModule):

    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf) 
        if self.hparams.task == 'optic_fovea':
            self.model = models.efficientnet_b4(num_classes=4)
        else:
            self.model = baseUNet(in_channel=conf['in_channel'], out_channel=conf['out_channel']) 
         
        
        self.total_loss = {'train':0, 'val':0, 'test':0} 
        if self.hparams.task == 'vessel':
            from .losses.dicebce import DiceBCELoss
            self.loss = DiceBCELoss()
            self.metrics = {'dice': Dice()} 
        elif self.hparams.task == 'optic_fovea':
            self.loss = torch.nn.MSELoss()
            self.metrics = {'optic': MSE(), 'fovea': MSE()}   

    def step(self, batch, batch_idx=None, tvt="train"):
        y_0 = batch['gt_image']
        if self.hparams.task == 'vessel':
            gt = batch['mask']  
            out = self.model(y_0)
            if tvt == 'train':
                loss = self.loss(out, gt) 
                return loss
            else:
                if batch_idx == 0:
                    self.visualize_out(y_0, gt, out)  
                self.update_metrics(pred=out.detach(), target=gt.detach()),  

        elif self.hparams.task == 'optic_fovea':
            gt = batch['gt']  
            out = self.model(y_0)
            out = out.sigmoid()
            if tvt == 'train':
                loss = self.loss(out, gt) 
                return loss

            else:
                if batch_idx == 0:
                    self.visualize_out(y_0, gt, out.detach())  
                self.update_metrics(pred=out.detach(), target=gt.detach()),  
 
         
    def visualize_out(self, img, gt, out):
        save_dir = os.path.join(self.hparams.save_dir, str(self.current_epoch))
        os.makedirs(save_dir, exist_ok=True) 
        if self.hparams.task == 'optic_fovea':
            res_im_m = []
            res_im_o = []
            for b in range(img.size(0)):
                im, m, o = img[b], gt[b].view(1, 2, 2), out[b].view(1, 2, 2)
                im_uint8 = ((im * 0.5 + 0.5) * 255).type(torch.uint8)
                res_im_m.append(draw_keypoints(im_uint8, m * 255, colors='#e5eb34', radius=5))
                res_im_o.append(draw_keypoints(im_uint8, o * 255, colors='#a530ab', radius=5))
            res = torch.cat([torch.stack(res_im_m), torch.stack(res_im_o)], dim=0) / 255
        elif self.hparams.task == 'vessel':
            res = torch.cat([img.mean(axis=1).unsqueeze(1), gt, out], dim=0)
        save_image(res.float(), 'a.jpg', nrow = gt.size(0), normalize=True)
        save_image(res.float(), save_dir + f'/{self.local_rank}_out.jpg', nrow = gt.size(0), normalize=True)  

    def update_metrics(self, pred, target): 
        if self.hparams.task == 'vessel':
            self.metrics['dice'].update(pred, target.int())
        elif self.hparams.task == 'optic_fovea':
            # [fovea_x, fovea_y, optic_x, optic_y]
            self.metrics['fovea'].update(pred[:,:2], target[:,:2])
            self.metrics['optic'].update(pred[:,2:], target[:,2:])
            

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
 