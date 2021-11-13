import json
import argparse
import os

def build_config(user_config, args):
    ## config
    config = get_default_config()
    for key, val in get_default_config().items():
        config[key]=getattr(args,key)
    config.update(user_config)
    build_config_(config)
    return config

def build_config_(config):
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
    config["model_py"] = "deepgrass.nn"
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


