
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

def test(config):

    model=networks[config["basic"]["model"]]

    pretrained_model_path=config["basic"]["pt_model_path"]
    device = config["basic"]["device"]
    exp_root_dir=config["basic"]["exp_root_path"]
    ds2_model_path=config["basic"]["ds2_model_path"]
    pt_model_path = config["basic"]["pt_model_path"]
    use_pt_model = config["basic"]["use_pt_model"]
    augmentation_config_name=config["basic"]["augmentation_config_name"]
    language_model_path=config["basic"]["language_model_path"]
    vocab_filepath=config["basic"]["vocab_filepath"]
    mean_std_filepath=config["basic"]["mean_std_filepath"]

    batch_size=config["test"]["batch_size"]
    max_duration=config["test"]["max_duration"],
    min_duration=config["test"]["min_duration"],
    segmented=config["test"]["segmented"]
    num_workers=config["test"]["num_workers"]

    test_csv= config["data"]["test_csv"]



    test_dataset = SpecgramGenerator(manifest=os.path.join(exp_root_dir, "data", test_csv),
                                      vocab_filepath=vocab_filepath,
                                      mean_std_filepath=mean_std_filepath,
                                      augmentation_config="{}",
                                      max_duration=max_duration,   #20,
                                      min_duration=min_duration, # 3
                                      segmented=segmented) # False


    dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=SpecgramGenerator.padding_batch)

    vocab_list = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    ds2_model = DeepSpeech2Model(model=model,
                                 ds2_model_path=ds2_model_path,
                                 vocab_list=vocab_list,
                                 device=device)

    if use_pt_model and pretrained_model_path:
        ds2_model.load_weights(pt_model_path)
    ds2_model.init_ext_scorer(1.4, 0.35, language_model_path)

    outputs = defaultdict(list)
    beam_alpha=1.1
    
    for i_batch, sample_batched in enumerate(dataloader):
        batch_results = ds2_model.infer_batch_probs(infer_data=sample_batched)
        batch_transcripts_beam = ds2_model.decode_batch_beam_search(
            probs_split=batch_results,
            beam_alpha=beam_alpha,
            beam_beta=0.35,
            beam_size=500,
            cutoff_prob=1.0,
            cutoff_top_n=40,
            num_processes=6)
        
        outputs["uttid"].extend(sample_batched["uttid"])
        outputs["probs"].extend(batch_results)
        outputs["asr"].extend(batch_transcripts_beam)
        outputs["text"].extend(sample_batched["trans"])

    df = pd.DataFrame.from_dict(outputs)
    return df

if __name__ == "__main__":
    from itertools import product
    import sys

    exp_yaml = sys.argv[1]
    saving_path = sys.argv[2]

    assert exp_yaml
    assert saving_path
    
    config = load_yaml_config(exp_yaml)

    result_df = test(config)
    result_df.to_pickle(os.path.join(saving_path, "test.pkl"))
