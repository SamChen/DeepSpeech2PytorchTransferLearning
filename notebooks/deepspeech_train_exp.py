
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



# def create_basic_documentation(config_path):
#     config = load_yaml_config(config_path)
#     return config, output_dir
networks = {"network.deepspeech_orig"         : network.deepspeech_orig,
            "network.deepspeech_newbottleneck": network.deepspeech_newbottleneck}

def main(config):

    model=networks[config["basic"]["model"]]
    #model=model

    pretrained_model_path=config["basic"]["pretrained_model_path"]
    lr=config["optimizer"]["learning_rate"]
    included_lr_key=config["optimizer"]["included_layer_keywords"]
    excluded_lr_key=config["optimizer"]["excluded_layer_keywords"]
    scheduler_gamma=config["scheduler"]["gamma"]
    exp_root_dir=config["basic"]["exp_root_path"]
    num_passes=config["basic"]["num_epochs_train"]
    num_iterations_print=config["basic"]["num_iterations_validate"]
    sorta_epoch=config["basic"]["num_sorta_epoch"]
    augmentation_config_path=config["basic"]["augmentation_config_path"]
    batch_size=config["basic"]["batch_size"]
    train_csv=config["data"]["train_csv"]
    val_csv=  config["data"]["val_csv"]
    test_csv= config["data"]["test_csv"]


    if augmentation_config_path:
        with open(os.path.join(exp_root_dir, "conf/augmentation.config"), 'r') as f:
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
                                      vocab_filepath="../models/baidu_en8k/vocab.txt",
                                      mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                      max_duration=20,
                                      min_duration=3,
                                      augmentation_config=augmentation_config,
                                      segmented=False)


    val_dataset = SpecgramGenerator(manifest=os.path.join(exp_root_dir, "data", val_csv),
                                    vocab_filepath="../models/baidu_en8k/vocab.txt",
                                    mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                    max_duration=30,
                                    min_duration=3,
                                    segmented=False)

    vocab_list = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    ds2_model = DeepSpeech2Model(model=model,
                                 vocab_list=vocab_list,
                                 pretrained_model_path="TBD",
                                 device="cuda")

    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(output_dir, "model_info.txt"), 'w') as f:
        f.write("DNN structure: \n{}\n".format(ds2_model.model))

    if pretrained_model_path:
        ds2_model.load_weights(pretrained_model_path)

    ds2_model.init_ext_scorer(1.4, 0.35, "../models/lm/common_crawl_00.prune01111.trie.klm")

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
        sorta_epoch=sorta_epoch)


if __name__ == "__main__":
    from itertools import product
    # use 1e-5 might reach to a good result if we train it long enough (i.e. 80 epoch)
    # lr= 2e-5, 1.7e-5# 4e-6# 7e-6
    # lr_key = ["deepspeech"]
    # lr = 7e-6
    # scheduler_gamma = 1# 0.95 # when learning rate is 5e-4
    # exclue_lr_key = None # ["bias"]
    # sorta_epoch=0 # number of epochs that use sorted batches

    # exp_root_dir="./iconect/models/pt_only_all_utt_wgt_transfer_mean"
    # train_csv = "train_short.csv"
    # val_csv   = "val_short.csv"
    # test_csv  = "test_short.csv"

    # # pretrained_model_path = "iconect/models/pt_only_all_utt_wgt_transfer_mean/exps/200304-12:06:03_lr7e-06-deepspeech/model_final.pth"
    # pretrained_model_path = None

    config = load_yaml_config("/home/samchen/temp/iconect/pt_only_all_utt_wgt_transfer/conf/experiment.yaml")


    # it looks like batch size is not the larger the better.
    main(config)
