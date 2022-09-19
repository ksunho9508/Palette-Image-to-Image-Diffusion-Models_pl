from .fundus_dm import FundusDM


def get_data_module(conf):
    dm_name = conf["DM_Name"]

    if dm_name in globals():
        return globals()[dm_name](conf)
    else:
        ValueError("[Trainer] {} is not implemented yet.".format(dm_name))
