import torch
import logging
import math
from deepgrass.model import SSM
from deepgrass.vssm import VariationalStateSpaceModel
import importlib


def build_model(config,dataset,device=None):
    input_dim = dataset.input_dim if dataset.input_dim is not None else 0
    state_dim = config["state_dim"]
    obs_dim = dataset.obs_dim
    obs_mask_enabled = dataset.obs_mask is not None
    
    nn_mod = importlib.import_module(config["model_py"])
    sys = VariationalStateSpaceModel(
            obs_dim =obs_dim, state_dim=state_dim, input_dim=input_dim,
            model_f =nn_mod.ModelF(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_h =nn_mod.ModelH(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_p0=nn_mod.ModelP0(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_q =nn_mod.ModelQ(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim,obs_mask_enabled=obs_mask_enabled),
            model_q0=nn_mod.ModelQ0(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim, obs_mask_enabled=obs_mask_enabled),
            delta_t=config["delta_t"],
            alpha={
                "recons":config["alpha_recons"],
                "temporal":config["alpha_temporal"],
                "beta":config["beta"],
            },
            device=device
            )

    model = SSM(config, sys, device=device)
    return model

