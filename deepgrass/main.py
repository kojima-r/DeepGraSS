#
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import logging

from six.moves import urllib
import numpy as np
import joblib
import json
import argparse

import deepgrass
from deepgrass.data_util import load_data
from deepgrass.plot_loss import plot_loss, plot_loss_detail
from deepgrass.model import SSM
from deepgrass.vssm import VariationalStateSpaceModel, ModelF, ModelH, ModelQ, ModelQ0, ModelP0

import torch
import torch.nn as nn
import torch.nn.functional as F


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


def build_config(config):
    # backward compatibility
    for key in config.keys():
        if "_npy" in config.keys():
            new_key = key.replace("_npy", "")
            config[new_key] = config[key]

    if "dim" in config:
        config["state_dim"] = config["dim"]
    if "result_path" in config:
        path = config["result_path"]
        os.makedirs(path, exist_ok=True)
        os.makedirs(path+"/model", exist_ok=True)
        os.makedirs(path+"/plot", exist_ok=True)
        os.makedirs(path+"/sim", exist_ok=True)
        config["save_model_path"] = path + "/model"
        config["save_result_test"] = path + "/test.jbl"
        config["save_result_train"] = path + "/train.jbl"
        config["simulation_path"] = path + "/sim"
        config["load_model"] = path + "/model/best.checkpoint"
        config["plot_path"] = path
        config["log_path"] = path


def get_default_config():
    config = {}
    # data and network
    # config["dim"]=None
    config["state_dim"] = 2
    # training
    config["epoch"] = 10
    config["patience"] = 5
    config["batch_size"] = 100
    #config["activation"] = "relu"
    #config["optimizer"] = "sgd"
    ##
    config["v_type"] = "single"
    config["system_scale"] = 0.1
    config["learning_rate"] = 1.0e-2
    # dataset
    config["train_valid_ratio"] = 0.2
    config["data_train"] = None
    config["data_test"] = None
    # save/load model
    config["save_model_path"] = None
    config["load_model"] = None
    config["save_result_train"] = None
    config["save_result_test"] = None
    config["save_result_filter"] = None

    config["delta_t"]=0.1
    config["alpha_recons"]=1.0
    config["alpha_temporal"]=1.0
    config["beta"]=1.0
    config["pretrain_epoch"] = 3
    
    config["weight_decay"] = 0.01
    config["hidden_layer_f"] = [32]
    config["hidden_layer_g"] = [32]
    config["hidden_layer_h"] = [32]
    """
    config["alpha"] = 1.0
    config["beta"] = 1.0
    config["gamma"] = 1.0
    ##
    config["curriculum_alpha"] = False
    config["curriculum_beta"] = False
    config["curriculum_gamma"] = False
    config["sampling_tau"] = 10  # 0.1
    config["normal_max_var"] = 5.0  # 1.0
    config["normal_min_var"] = 1.0e-5
    config["zero_dynamics_var"] = 1.0
    config["pfilter_sample_size"] = 10
    config["pfilter_proposal_sample_size"] = 1000
    config["pfilter_save_sample_num"] = 100
    config["label"] = "multinominal"
    config["task"] = "generative"
    # config["state_type"]="discrete"
    config["state_type"] = "normal"
    config["sampling_type"] = "none"
    config["time_major"] = True
    config["steps_train_npy"] = None
    config["steps_test_npy"] = None
    config["sampling_type"] = "normal"
    config["emission_type"] = "normal"
    config["state_type"] = "normal"
    config["dynamics_type"] = "distribution"
    config["pfilter_type"] = "trained_dynamics"
    config["potential_enabled"] = (True,)
    config["potential_grad_transition_enabled"] = (True,)
    config["potential_nn_enabled"] = (False,)
    config["potential_grad_delta"] = 0.1
    #
    config["field_grid_num"] = 30
    config["field_grid_dim"] = None
    """
    # generate json
    return config

def run_train_mode(config, logger):
    logger.info("... loading data")
    all_data = load_data(mode="train", config=config, logger=logger)
    train_data, valid_data = all_data.split(1.0 - config["train_valid_ratio"])
    print("train_data_size:", train_data.num)

    # defining dimensions from given data
    print("observation dimension:", train_data.obs_dim)
    print("input dimension:", train_data.input_dim)
    print("state dimension:", train_data.state_dim)
    input_dim = train_data.input_dim if train_data.input_dim is not None else 0
    state_dim = config["state_dim"]
    obs_dim = train_data.obs_dim
    #
    if torch.cuda.is_available():
        device = 'cuda'
        print("device: cuda")
    else:
        device = 'cpu'
        print("device: cpu")
    # defining system
    sys = VariationalStateSpaceModel(
            obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim,
            model_f=ModelF(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_h=ModelH(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_p0=ModelP0(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_variational=ModelQ(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_variational0=ModelQ0(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            delta_t=config["delta_t"],
            alpha={
                "recons":config["alpha_recons"],
                "temporal":config["alpha_temporal"],
                "beta":config["beta"],
            },
            device=device
            )
    # training NN from data
    model = SSM(config, sys, device=device)
    train_loss,valid_loss,flag = model.fit(train_data, valid_data)

    joblib.dump(train_loss, config["save_model_path"]+"/train_loss.pkl")
    joblib.dump(valid_loss, config["save_model_path"]+"/valid_loss.pkl")
    plot_loss(train_loss,valid_loss,config["plot_path"]+"/loss.png")
    plot_loss_detail(train_loss,valid_loss,config["plot_path"]+"/loss_detail.png")

    model.load_ckpt(config["save_model_path"]+"/best.checkpoint")
    loss, states, obs_gen = model.simulate_with_data(valid_data)
    save_simulation(config,valid_data,states,obs_gen)
    joblib.dump(loss, config["simulation_path"]+"/last_loss.pkl")
    
    
def save_simulation(config,data,states,obs_gen):
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/obs.npy"
        print("[SAVE]", filename)
        np.save(filename, data.obs)
        if data.state is not None:
            filename=config["simulation_path"]+"/state_true.npy"
            print("[SAVE]", filename)
            np.save(filename, data.state)
        if data.input is not None:
            filename=config["simulation_path"]+"/input.npy"
            print("[SAVE]", filename)
            np.save(filename, data.input)
        filename=config["simulation_path"]+"/obs_gen.npy"
        print("[SAVE]", filename)
        np.save(filename, obs_gen.to("cpu").detach().numpy().copy())
        filename=config["simulation_path"]+"/states.npy"
        print("[SAVE]", filename)
        np.save(filename, states.to("cpu").detach().numpy().copy())


def run_pred_mode(config, logger):
    logger.info("... loading data")
    all_data = load_data(mode="test", config=config, logger=logger)
    print("data_size:", all_data.num)

    # defining dimensions from given data
    print("observation dimension:", all_data.obs_dim)
    print("input dimension:", all_data.input_dim)
    print("state dimension:", all_data.state_dim)
    input_dim = all_data.input_dim if all_data.input_dim is not None else 1
    state_dim = config["state_dim"]
    obs_dim = all_data.obs_dim
    #
    if torch.cuda.is_available():
        device = 'cuda'
        print("device: cuda")
    else:
        device = 'cpu'
        print("device: cpu")
    # defining system
    sys = VariationalStateSpaceModel(
            obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim,
            model_f=ModelF(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_h=ModelH(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_p0=ModelP0(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_variational=ModelQ(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            model_variational0=ModelQ0(obs_dim=obs_dim, state_dim=state_dim, input_dim=input_dim),
            delta_t=config["delta_t"],
            alpha={
                "recons":config["alpha_recons"],
                "temporal":config["alpha_temporal"],
                "beta":config["beta"],
            },
            device=device
            )
    # loading NN from data
    model = SSM(config, sys, device=device)
    model.load_ckpt(config["load_model"])
    logger.info("... simulating data")
    loss, states, obs_gen = model.simulate_with_data(all_data)
    save_simulation(config,all_data,states,obs_gen)
    obs_gen=obs_gen.to("cpu").detach().numpy().copy()
    ##
    x=np.sum((all_data.obs-obs_gen)**2,axis=2)
    x=np.mean(x,axis=1)
    mse=np.mean(x,axis=0)
    logger.info("... loading data")
    logger.info("mean error: {}".format(mse))
    ####
    print("=== field")
    pt,vec=model.get_vector_field(state_dim, dim=[0,1],min_v=-3,max_v=3,delta=0.2)
    if "simulation_path" in config:
        os.makedirs(config["simulation_path"], exist_ok=True)
        filename=config["simulation_path"]+"/field_pt.npy"
        print("[SAVE]", filename)
        np.save(filename, pt)
        filename=config["simulation_path"]+"/field_vec.npy"
        print("[SAVE]", filename)
        np.save(filename, vec)
    ####

def set_file_logger(logger,config,filename):
    if "log_path" in config:
        filename=config["log_path"]+"/"+filename
        h = logging.FileHandler(filename=filename, mode="w")
        h.setLevel(logging.INFO)
        logger.addHandler(h)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/infer")
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--save_config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument("--no-config", action="store_true", help="use default setting")
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument(
        "--hyperparam",
        type=str,
        default=None,
        nargs="?",
        help="hyperparameter json file",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )
    parser.add_argument("--profile", action="store_true", help="")
    ## config
    for key, val in get_default_config().items():
        if type(val) is int:
            parser.add_argument("--"+key, type=int, default=val, help="[config integer]")
        elif type(val) is float:
            parser.add_argument("--"+key, type=float, default=val, help="[config float]")
        elif type(val) is bool:
            parser.add_argument("--"+key, type=bool, default=val, help="[config float]")
            #parser.add_argument("--"+key, action="store_true", help="[config bool]")
        else:
            parser.add_argument("--"+key, type=str, default=val, help="[config string]")
    args = parser.parse_args()
    # config
    config = get_default_config()
    for key, val in get_default_config().items():
        config[key]=getattr(args,key)
    # 
    if args.config is None:
        if not args.no_config:
            parser.print_help()
            quit()
    else:
        print("[LOAD]",args.config)
        fp = open(args.config, "r")
        config.update(json.load(fp))
    build_config(config)
    # gpu/cpu
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # profile
    config["profile"] = args.profile
    #
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")

    # setup
    mode_list = args.mode.split(",")
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    for mode in mode_list:
        # mode
        if mode == "train":
            set_file_logger(logger,config,"log_train.txt")
            run_train_mode(config, logger)
        elif mode == "infer" or mode == "test":
            set_file_logger(logger,config,"log_test.txt")
            if args.model is not None:
                config["load_model"] = args.model
            run_pred_mode(config, logger)
        elif mode == "filter":
            if args.model is not None:
                config["load_model"] = args.model
            filtering(sess, config)
        elif mode == "filter_discrete":
            filter_discrete_forward(sess, config)
        elif mode == "train_fivo":
            train_fivo(sess, config)
        elif mode == "field":
            field(sess, config)
        elif mode == "potential":
            potential(sess, config)
        elif args.mode == "filter_server":
            filtering_server(sess, config=config)

    if args.save_config is not None:
        print("[SAVE] config: ", args.save_config)
        fp = open(args.save_config, "w")
        json.dump(
            config,
            fp,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            cls=NumPyArangeEncoder,
        )


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
