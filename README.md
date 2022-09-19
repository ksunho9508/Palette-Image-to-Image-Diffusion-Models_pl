# FAI (updated by 220831)
## Step1: coarse training (10 epochs)
> explanation
- Just coarse training (with entropy min to seg loss)

> python code
```python
python main.py --config configs/fai_coarse_training.yaml
``` 
> config details
- only_test: if True, conf['resume'] should be specified as the trained model's ckpt directory
- selection_criteria: comprehensive average method for auroc & auprc & fpr
- cls_loss: BCE
- seg_loss: NLL + entropy_min
***
## Step2: pseudo mask generating (1 epoch)
> explanation
- This step is for generating pseudo mask for fine_training (step3). The save masks have float values (not binary) because threshold operator can be adopted at step3.

> python code
```python
python main.py --config configs/fai_ten_pseudo_mask.yaml
``` 
> config details
- pseudo_mask_dir = save_dir
- only_test should be True! (no aug, no loss backward)
- resume: step1's best model dir (e.g., outputs/fai_coarse_training/checkpoints/1-auroc0.965-auprc0.528-fpr0.167.ckpt)

***
## Step3: fine training (with label correct or not) (20 epochs)
> explanation
- fine_training (with pseudo mask generated in step2)

> python code 
```python
python main.py --config configs/fai_coarse_training.yaml
``` 
> python code with label correct
```python
python main.py --config configs/fai_coarse_training_mtLC.yaml
``` 
> config details 
- only_test: if True, conf['resume'] should be specified as the trained model's ckpt directory
- selection_criteria: comprehensive average method for auroc & auprc & fpr
- cls_loss: BCE
- seg_loss: BCE
- pseudo_mask: saved pseudo_mask directory
- pseudo_mask_threshold: float in (0,1) 
- lc_method: choose option among of [VanilaLC, ProSelfLC, MultiSelfLC]
***
## Measure performance
> explanation
- The above step with 'only_test == True' should be executed in advacne.
- Then, the saved file, 'test_preds.csv', will be used to measure performance in 'measure_performace_from_preds.ipynb'.

> Example result

*  coarse training: [config](https://github.com/vuno/FUNDUS_FAI_DEV/blob/fai/configs/fai_coarse_training.yaml)

|            |     AVG | Hemorrhage | HardExudate |      CWP |   Drusen | VascularAbnormality | Membrane | ChroioretinalAtrophy | MyelinatedNerveFiber | RNFLDefect | GlaucomatousDiscChange | NonGlaucomatousDiscChange | MacularHole |
|-----------:|--------:|-----------:|------------:|---------:|---------:|--------------------:|---------:|---------------------:|---------------------:|-----------:|-----------------------:|--------------------------:|------------:|
|      AUROC | 0.99139 |   0.996671 |    0.997095 | 0.997922 | 0.987941 |            0.991497 | 0.996513 |             0.997220 |             0.999968 |   0.972805 |               0.982543 |                  0.981925 |    0.994631 |
|    F1 best | 0.79818 |   0.918367 |    0.926316 | 0.784091 | 0.839169 |            0.684932 | 0.879339 |             0.876543 |             0.977778 |   0.614907 |               0.738019 |                  0.538745 |    0.800000 |
| prec@tpr85 | 0.66981 |   0.977157 |    0.985646 | 0.655462 | 0.805740 |            0.328042 | 0.899329 |             0.890728 |             1.000000 |   0.344086 |               0.526022 |                  0.216518 |    0.409091 |
| prec@tpr95 | 0.51163 |   0.836576 |    0.871212 | 0.505814 | 0.654370 |            0.147992 | 0.674157 |             0.732360 |             1.000000 |   0.110086 |               0.275261 |                  0.141927 |    0.189873 |
| prec@tpr99 | 0.26621 |   0.340944 |    0.204604 | 0.328520 | 0.421836 |            0.045090 | 0.355353 |             0.435327 |             0.758621 |   0.051205 |               0.142919 |                  0.085671 |    0.024545 |

* fine training: [config](https://github.com/vuno/FUNDUS_FAI_DEV/blob/fai/configs/fai_fine_training.yaml)

|            | AVG     | Hemorrhage | HardExudate |      CWP |   Drusen | VascularAbnormality | Membrane | ChroioretinalAtrophy | MyelinatedNerveFiber | RNFLDefect | GlaucomatousDiscChange | NonGlaucomatousDiscChange | MacularHole |
|-----------:|---------|-----------:|------------:|---------:|---------:|--------------------:|---------:|---------------------:|---------------------:|-----------:|-----------------------:|--------------------------:|------------:|
|      AUROC | 0.99148 |   0.995882 |    0.997334 | 0.997660 | 0.988309 |            0.989387 | 0.996804 |             0.997328 |             0.999959 |   0.972371 |               0.983942 |                  0.983172 |    0.995659 |
|    F1 best |  0.8022 |   0.925339 |    0.933054 | 0.801980 | 0.837558 |            0.677419 | 0.876190 |             0.888179 |             0.936170 |   0.663185 |               0.750769 |                  0.549763 |    0.786885 |
| prec@tpr85 | 0.68836 |   0.969773 |    0.980952 | 0.742857 | 0.821147 |            0.413333 | 0.890365 |             0.908784 |             0.950000 |   0.360360 |               0.559289 |                  0.266484 |    0.397059 |
| prec@tpr95 | 0.48845 |   0.812854 |    0.821429 | 0.465241 | 0.648133 |            0.123457 | 0.689655 |             0.741379 |             0.913043 |   0.139735 |               0.327461 |                  0.112255 |    0.066815 |
| prec@tpr99 | 0.26605 |   0.266191 |    0.268156 | 0.229798 | 0.379803 |            0.036500 | 0.410526 |             0.422973 |             0.846154 |   0.052855 |               0.142055 |                  0.086856 |    0.050736 |

* fine training with mtLC: [config](https://github.com/vuno/FUNDUS_FAI_DEV/blob/fai/configs/fai_fine_training_mtLC.yaml)


|            |     AVG | Hemorrhage | HardExudate |      CWP |   Drusen | VascularAbnormality | Membrane | ChroioretinalAtrophy | MyelinatedNerveFiber | RNFLDefect | GlaucomatousDiscChange | NonGlaucomatousDiscChange | MacularHole |
|-----------:|--------:|-----------:|------------:|---------:|---------:|--------------------:|---------:|---------------------:|---------------------:|-----------:|-----------------------:|--------------------------:|------------:|
|      AUROC | 0.99223 |   0.996714 |    0.997359 | 0.997641 | 0.988589 |            0.993312 | 0.996541 |             0.996646 |             0.999949 |   0.977363 |               0.982681 |                  0.983986 |    0.996013 |
|    F1 best | 0.80410 |   0.932735 |    0.931106 | 0.789744 | 0.844363 |            0.708333 | 0.878049 |             0.891339 |             0.933333 |   0.658537 |               0.749621 |                  0.557940 |    0.774194 |
| prec@tpr85 | 0.70781 |   0.979644 |    0.985646 | 0.709091 | 0.831435 |            0.446043 | 0.905405 |             0.887789 |             1.000000 |   0.419948 |               0.603411 |                  0.282799 |    0.442623 |
| prec@tpr95 | 0.49105 |   0.808271 |    0.871212 | 0.441624 | 0.673823 |            0.114379 | 0.659341 |             0.728814 |             0.875000 |   0.166357 |               0.278169 |                  0.100184 |    0.175439 |
| prec@tpr99 | 0.25997 |   0.430769 |    0.235294 | 0.232143 | 0.366695 |            0.069923 | 0.403101 |             0.311443 |             0.758621 |   0.049445 |               0.140598 |                  0.085541 |    0.036131 |
***
### Conda Environment
- conda create -n pl python=3.9
- conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- pip install pytorch-lightning
- pip install opencv-python
- conda install pandas
- pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
- pip install git+https://github.com/ildoonet/pytorch-randaugment