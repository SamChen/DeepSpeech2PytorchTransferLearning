#!/usr/bin/env python

import os
import sys
import pandas as pd
from datetime import datetime
from collections import defaultdict
sys.path.append("..")

from model_utils.model import DeepSpeech2Model
import model_utils.network as network
from data_utils.dataloader import SpecgramGenerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(network, lr, lr_key, num_passes=40,
         num_iterations_print=400, augmentation_config="{}",
         batch_size=32,
         output_root_dir="./iconect/models/pt_only_all_utt/",
         train_csv="train.csv",
         val_csv="val.csv",
         test_csv="test.csv"):

    train_dataset = SpecgramGenerator(manifest=os.path.join(output_root_dir, train_csv),
                                      vocab_filepath="../models/baidu_en8k/vocab.txt",
                                      mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                      max_duration=float('inf'),
                                      min_duration=3.0,
                                      augmentation_config=augmentation_config,
                                      segmented=False)

    val_dataset = SpecgramGenerator(manifest=os.path.join(output_root_dir, val_csv),
                                    vocab_filepath="../models/baidu_en8k/vocab.txt",
                                    mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                    max_duration=float('inf'),
                                    min_duration=3.0,
                                    segmented=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=4,
                                  collate_fn=SpecgramGenerator.padding_batch)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=3,
                                  collate_fn=SpecgramGenerator.padding_batch)

    vocab_list = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    ds2_model = DeepSpeech2Model(
        network=network,
        vocab_size=28,
        pretrained_model_path="TBD",
        device="cuda")


    ds2_model.init_ext_scorer(1.4, 0.35, "../models/lm/common_crawl_00.prune01111.trie.klm", vocab_list=vocab_list)


    filename = datetime.now().strftime("lr{}-{}_%y%m%d-%H:%M:%S".format(lr, "_".join(lr_key)))
    output_dir = os.path.join(output_root_dir, "exps", filename)
    log_dir=os.path.join(output_root_dir, "tensorboard", filename)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    ds2_model.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                    lr_key=lr_key,
                    learning_rate=lr,
                    gradient_clipping=400,
                    num_passes=num_passes,
                    num_iterations_print=num_iterations_print,
                    writer=tensorboard_writer,
                    output_dir=output_dir)

    outputs = defaultdict(list)
    #TODO: make it as a function for DeepSpeech Class
    for i_batch, sample_batched in enumerate(val_dataloader):
        batch_results = ds2_model.infer_batch_probs(infer_data=sample_batched)
        batch_transcripts_beam = ds2_model.decode_batch_beam_search(probs_split=batch_results,
                                                                     beam_alpha=1.4,
                                                                     beam_beta=0.35,
                                                                     beam_size=500,
                                                                     cutoff_prob=1.0,
                                                                     cutoff_top_n=40,
                                                                     vocab_list=vocab_list,
                                                                     num_processes=5)

        outputs["uttid"].extend(sample_batched["uttid"])
        outputs["probs"].extend(batch_results)
        outputs["asr"].extend(batch_transcripts_beam)
        outputs["text"].extend(sample_batched["trans"])

    df = pd.DataFrame.from_dict(outputs)

    df.to_pickle(os.path.join(output_root_dir, "decoded", "deepspeech_val_{}.pkl".format(filename)))

if __name__ == "__main__":
    from itertools import product

    network=network.deepspeech_LocalDotAtten
    # network=deepspeech_orig
    lr=5e-5# 7e-6
    lr_key = ["deepspeech_exp_layers"]

    output_root_dir="./iconect/models/pt_only_all_utt_sliced4"
    train_csv = "data/train_short.csv"
    val_csv   = "data/val_short.csv"
    test_csv  = "data/test_short.csv"

    with open(os.path.join(output_root_dir, "conf/augmentation.config"), 'r') as f:
        augmentation_config = f.read()

    main(network=network,
         lr=lr, lr_key=lr_key,
         num_passes=40, num_iterations_print=200,
         augmentation_config=augmentation_config,
         batch_size=16,
         output_root_dir=output_root_dir,
         train_csv=train_csv,
         val_csv=val_csv,
         test_csv=test_csv)

