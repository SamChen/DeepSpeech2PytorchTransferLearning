#!/usr/bin/env python

import os
import sys
import pandas as pd
from datetime import datetime
from distutils.dir_util import mkpath
import shutil
from collections import defaultdict
sys.path.append("..")

from model_utils.model import DeepSpeech2Model
from utils.yaml_loader import load_yaml_config
import model_utils.network as network
from data_utils.dataloader import SpecgramGenerator

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml as yaml

networks = {"network.deepspeech_orig"         : network.deepspeech_orig,
            "network.deepspeech_newbottleneck": network.deepspeech_newbottleneck}

def main(config):

    model=networks[config["basic"]["model"]]

    pretrained_model_path=config["basic"]["pt_model_path"]
    device = config["basic"]["device"]
    exp_root_dir=config["basic"]["exp_root_path"]
    ds2_model_path=config["basic"]["ds2_model_path"]
    augmentation_config_name=config["basic"]["augmentation_config_name"]
    language_model_path=config["basic"]["language_model_path"]
    vocab_filepath=config["basic"]["vocab_filepath"]
    mean_std_filepath=config["basic"]["mean_std_filepath"]

    batch_size=config["train"]["batch_size"]
    max_duration=config["train"]["max_duration"]
    min_duration=config["train"]["min_duration"]
    segmented=config["train"]["segmented"]
    num_passes=config["train"]["num_total_epochs"]
    num_iterations_print=config["train"]["num_iterations_validate"]
    sortN_epoch=config["train"]["num_sorted_epoch"]
    num_workers=config["train"]["num_workers"]

    
    print(num_workers)
    
    
    # max_duration=config["test"]["max_duration"],
    # min_duration=config["test"]["min_duration"],
    # batch_size=config["test"]["batch_size"]
    # max_duration=config["test"]["max_duration"]
    # min_duration=config["test"]["min_duration"]
    # segmented=config["test"]["segmented"]
    # num_workers=config["test"]["num_workers"]

    train_csv=config["data"]["train_csv"]
    val_csv=  config["data"]["val_csv"]
    test_csv= config["data"]["test_csv"]

    lr=config["optimizer"]["learning_rate"]
    included_lr_key=config["optimizer"]["included_layer_keywords"]
    excluded_lr_key=config["optimizer"]["excluded_layer_keywords"]

    try:
        specific_lr_dict = config["optimizer"]["specific_lr_dict"]
    except:
        specific_lr_dict = None
        Warning("You miss the keyword specific_lr_dict")

    scheduler_gamma=config["scheduler"]["gamma"]

    if augmentation_config_name:
        with open(os.path.join(exp_root_dir, "conf",augmentation_config_name), 'r') as f:
            augmentation_config = f.read()
    else:
        augmentation_config = "{}"


    filename = datetime.now().strftime("%y%m%d-%H:%M:%S")
    output_dir = os.path.join(exp_root_dir, "exps", filename)
    mkpath(os.path.join(output_dir, "models"))
    mkpath(os.path.join(output_dir, "vals"))
    # shutil.copy2(config_path, os.path.join(output_dir, "experiment.yaml"))
    with open(os.path.join(output_dir, "experiment.yaml"), 'w') as f:
        yaml.safe_dump(config, stream=f)

    log_dir=os.path.join(exp_root_dir, "tensorboard", filename)



    train_dataset = SpecgramGenerator(manifest=os.path.join(exp_root_dir, "data", train_csv),
                                      vocab_filepath=vocab_filepath,
                                      mean_std_filepath=mean_std_filepath,
                                      augmentation_config=augmentation_config,
                                      max_duration=max_duration,   #20,
                                      min_duration=min_duration, # 3
                                      segmented=segmented) # False


    val_dataset = SpecgramGenerator(manifest=os.path.join(exp_root_dir, "data", val_csv),
                                    vocab_filepath=vocab_filepath,
                                    mean_std_filepath=mean_std_filepath,
                                    max_duration=max_duration,   #20,
                                    min_duration=min_duration, # 3
                                    segmented=segmented) # False

    vocab_list = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    ds2_model = DeepSpeech2Model(model=model,
                                 ds2_model_path=ds2_model_path,
                                 vocab_list=vocab_list,
                                 device=device)

    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(output_dir, "model_info.txt"), 'w') as f:
        f.write("DNN structure: \n{}\n".format(ds2_model.model))

    if pretrained_model_path:
        ds2_model.load_weights(pretrained_model_path)

    ds2_model.init_ext_scorer(1.4, 0.35, language_model_path)

    ds2_model.train(
        train_dataset=train_dataset,
        train_batchsize=batch_size,
        val_dataset=val_dataset,
        val_batchsize=batch_size,
        collate_fn=SpecgramGenerator.padding_batch,
        lr_key=included_lr_key,
        exclue_lr_key=excluded_lr_key,
        learning_rate=lr,
        scheduler_gamma=scheduler_gamma,
        gradient_clipping=40,
        num_passes=num_passes,
        num_iterations_print=num_iterations_print,
        writer=tensorboard_writer,
        output_dir=output_dir,
        sortN_epoch=sortN_epoch,
        num_workers=num_workers,
        specific_lr_dict=specific_lr_dict)


if __name__ == "__main__":
    from itertools import product
    import sys

    exp_yaml = sys.argv[1]
    if not exp_yaml:
        exp_yaml = "example/conf/experiment.yaml"
    
    config = load_yaml_config(exp_yaml)
    # it looks like batch size is not the larger the better.
    main(config)
