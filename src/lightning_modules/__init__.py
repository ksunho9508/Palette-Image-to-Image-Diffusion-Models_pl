from .diffusion_lm import DiffusionLM  
from .de_diff_lm import DE_DiffLM
from .regression_lm import DE_RegressionLM
from .downstream_lm import DownstreamLM
def get_lightning_module(conf):
    lm_name = conf["LM_Name"]

    if lm_name in globals():
        return globals()[lm_name](conf)
    else:
        ValueError("[Trainer] {} is not implemented yet.".format(lm_name))

 