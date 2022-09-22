import os
import torch
from torch import optim  
from torchvision.utils import save_image
from pytorch_lightning import LightningModule
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts 
from .models import DiffusionModel, InfoMax_DiffusionModel 
from .metrics import * 
from scipy.stats import gmean
 
class DiffusionLM(LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        if conf['model_name'] == 'InfoMax_DiffusionModel':
            self.model = InfoMax_DiffusionModel(conf, device = self.device)  
        else:
            self.model = DiffusionModel(conf, device = self.device) 
            
        if 'pretrain_weight' in conf:
            task = conf['task']
            self.load_from_pretrain(os.path.join(conf['pretrain_weight'], f'{task}.pth')) 
        self.debug = True if conf['debug'] else False
        self.total_loss = {'train':0, 'val':0, 'test':0} 
        self.metrics = {'fid': FID().to(self.device), 
                        'is': IS().to(self.device), 
                        'psnr': PSNR().to(self.device), 
                        'ssim': SSIM().to(self.device)
                        } 

    def load_from_pretrain(self, pretrain_dir):
        self.model.set_new_noise_schedule(phase='train', device=self.device)
        self.model.load_state_dict(torch.load(pretrain_dir, map_location=self.device), strict=False)
        print('loaded pretrained weight')

    def step(self, batch, batch_idx=None, tvt="train"):
        y_0 = batch['gt_image']
        y_cond = batch['cond_image']
        if 'mask' in batch:
            m = batch['mask']
        else:
            m = None
        if tvt == 'val' and self.debug:
            save_image(batch['gt_image'], 'gt_image.jpg', normalize=True) 
            save_image(batch['cond_image']*0.5 + 0.5, 'cond_image.jpg')  
        

        if tvt == 'train':
            loss = self.model(y_0, y_cond, mask=m, device=self.device)
            return  loss
        else:
            y_result = self.model.restoration(
                y_cond=y_cond, 
                y_t=y_cond,
                y_0=y_0,
                mask=m,
                sample_num=8, 
                device=self.device,
                only_final=True if batch_idx!= 0 else False)

            if batch_idx == 0:
                self.visualize_restoration(y_0, y_cond, y_result)    
                y_result = y_result[-1 * y_0.size(0):] # extract only final
                
            self.update_metrics(pred=y_result.detach(), target=y_0.detach()) 
     
    
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

    def on_train_start(self) -> None:
        self.model.set_new_noise_schedule(phase='train', device=self.device)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, tvt="train") 
        if self.global_rank == 0 and batch_idx % self.hparams.sample_iter == 0:
            self.logging_loss(loss, tvt='train', on_step=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"])

        return {"loss": loss}

    def on_validation_start(self):
        if self.debug:
            self.model.set_new_noise_schedule(phase='debug', device=self.device) 
        else:
            self.model.set_new_noise_schedule(phase='val', device=self.device) 

        for m in self.metrics.values():
            m = m.to(self.device)

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx=batch_idx, tvt="val")    

    def on_validation_epoch_end(self) -> None: 
        self.model.set_new_noise_schedule(phase='train', device=self.device) 
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
 