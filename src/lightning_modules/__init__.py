from .diffusion_lm import DiffusionLM 


def get_lightning_module(conf):
    lm_name = conf["LM_Name"]

    if lm_name in globals():
        return globals()[lm_name](conf)
    else:
        ValueError("[Trainer] {} is not implemented yet.".format(lm_name))

 