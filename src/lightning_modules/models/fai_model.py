from torch import nn
from torchvision import models  
Finding_target = [
    "Hemorrhage",
    "HardExudate",
    "CWP",
    "Drusen",
    "VascularAbnormality",
    "Membrane",
    "ChroioretinalAtrophy",
    "MyelinatedNerveFiber",
    "RNFLDefect",
    "GlaucomatousDiscChange",
    "NonGlaucomatousDiscChange",
    "MacularHole",
]
Disease_target = [
    'EpiretinalMembrane',
    'GlaucomaSuspect',
    'WetAMD',
    'DryAMD',
    'CRVO',
    'BRVOHemiCRVO',
    'AdvancedDR',
    'EarlyDR']


class FAIModel(nn.Module):
    def __init__(self, task='finding'):
        super().__init__()
        if task == 'finding':
            self.target = Finding_target
        elif task == 'disease':
            self.target = Disease_target
        self.network = getattr(models, "efficientnet_b4")(
            weights="EfficientNet_B4_Weights.IMAGENET1K_V1"
        )
        in_channels = self.network.classifier[-1].in_features

        self.segmentor = nn.ModuleDict(
            {t: nn.Conv2d(in_channels, 1, (1, 1)) for t in self.target}
        )
        # self.branch_network(in_channels=in_channels, branch_mode=conf["branch_mode"])
        self.pooling_method = 'single'
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)

        delattr(self.network, "classifier")
        delattr(self.network, "avgpool")

    def forward(self, x):
        x = self.network.features(x)

        seg = {t: self.segmentor[t](x) for t in self.target}
        cls = {
            t: (self._avg_pool(seg[t]) + self._max_pool(seg[t])).squeeze()
            for t in self.target
        }

        return cls, seg

    def get_target(self):
        return self.target
 

 