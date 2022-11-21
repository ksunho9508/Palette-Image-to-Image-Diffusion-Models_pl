import os
import torch
from torch import optim  
from torchvision.utils import save_image
from pytorch_lightning import LightningModule
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts 
from .models import DiffusionModel
from .metrics import * 
from scipy.stats import gmean
 
class DiffusionLM(LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf) 
        self.model = DiffusionModel(conf, device = self.device) 
            
        if 'pretrain_weight' in conf: 
            # self.load_from_pretrain(conf['pretrain_weight']) 
            task = conf['task']
            self.load_from_pretrain(os.path.join(conf['pretrain_weight'], f'{task}.pth')) 
        self.debug = True if conf['debug'] else False
        self.total_loss = {'train':0, 'val':0, 'test':0} 
        self.metrics = {'fid': FID().to(self.device), 
                        # 'is': IS().to(self.device), 
                        'psnr': PSNR().to(self.device), 
                        'ssim': SSIM().to(self.device)
                        } 
        
    # def load_from_pretrain(self, ckpt_path): 
    #     checkpoint = torch.load(ckpt_path, map_location=self.device)
    #     state_dict = checkpoint['state_dict']
    #     for key in list(state_dict.keys()):
    #         if 'model.' in key:
    #             state_dict[key.replace('model.', '')] = state_dict[key]
    #             del state_dict[key] 
    #     self.model.load_state_dict(state_dict, strict=False) 
    #     print('loaded pretrained weight')

    def load_from_pretrain(self, pretrain_dir):
        self.model.set_new_noise_schedule(phase='train', device=self.device)
        self.model.load_state_dict(torch.load(pretrain_dir, map_location=self.device), strict=False)
        print('loaded pretrained weight')


    def step(self, batch, batch_idx=None, tvt="train"):
        y_0 = batch['gt_image'] 
        # y_cond = batch['cond_image'] 
        y_cond = torch.randn_like(y_0)

        m = de_mask = None
        if 'mask' in batch:
            m = batch['mask'] 
        if 'de_mask' in batch:
            de_mask = batch['de_mask'].float()

        if tvt == 'val' and self.debug:
            save_image(batch['gt_image'], 'gt_image.jpg', normalize=True) 
            # save_image(batch['cond_image']*0.5 + 0.5, 'cond_image.jpg')  

        if tvt == 'train':
            loss = self.model(y_0, y_cond, device=self.device) #mask=m.unsqueeze(1), ) 
            return  loss
        else:
            if tvt == 'val':
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
                if not self.trainer.sanity_checking:
                    self.update_metrics(pred=y_result.detach(), target=y_0.detach()) 
            else:
                batch_size = y_cond.size(0)
                y_cond = torch.cat([y_cond, y_cond, y_cond, y_cond])
                y_0 = torch.cat([y_0, y_0, y_0, y_0]) 
                m = torch.cat([m, m, m, m]) 
                y_result = self.model.restoration(
                    y_cond=y_cond, 
                    y_t=y_cond,
                    y_0=y_0,
                    mask=m,
                    sample_num=8, 
                    device=self.device,
                    only_final=True) 

                # save result
                for b in range(batch_size):
                    y_0_ = y_0[b].unsqueeze(0) 
                    save_image(y_0_, os.path.join('finding_inpainting_results', f'{self.hparams.mask_mode}', 'origin', batch['path'][b]), normalize=True) 
                    # y_cond_ = y_cond[b].unsqueeze(0) 
                    # save_image(y_cond_, os.path.join('finding_inpainting_results', 'y_cond', batch['path'][b]), normalize=True)

                    for k in range(1, 5):
                        y_hat_ = y_result[(k-1)*batch_size + b].unsqueeze(0) 
                        save_image(y_hat_, os.path.join('finding_inpainting_results', f'{self.hparams.mask_mode}', str(k), batch['path'][b]), normalize=True)

    def update_metrics(self, pred, target):
        for m in ['psnr', 'ssim']:
            self.metrics[m].update(pred, target)       

        pred_uint8 = ((pred * 0.5 + 0.5) * 255).type(torch.uint8) 
        target_uint8 = ((target * 0.5 + 0.5) * 255).type(torch.uint8) 
        # for m in ['is']:
        #     self.metrics[m].update(pred_uint8)        
        for m in ['fid']:
            self.metrics[m].update(target_uint8, real=True)
            self.metrics[m].update(pred_uint8, real=False)
    
    def compute_metrics(self, tvt='val'):
        tot = []
        for n, m in self.metrics.items():
            res = m.compute()
            # if n == 'is':
            #     res = res[0]
            # print(n, m, res)
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
        if not self.trainer.sanity_checking:
            self.compute_metrics()

    def on_test_start(self): 
        if self.debug:
            self.model.set_new_noise_schedule(phase='debug', device=self.device) 
        else:
            self.model.set_new_noise_schedule(phase='test', device=self.device)  

        os.makedirs(os.path.join('finding_inpainting_results', f'{self.hparams.mask_mode}', 'origin'), exist_ok=True) 
        for k in range(1,5):
            save_dir = os.path.join('finding_inpainting_results', f'{self.hparams.mask_mode}', str(k))
            os.makedirs(save_dir, exist_ok=True) 

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx=batch_idx, tvt="test")    

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
 