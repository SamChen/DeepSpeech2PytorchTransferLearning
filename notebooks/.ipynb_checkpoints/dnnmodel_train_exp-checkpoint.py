#!/usr/bin/env python

import sys
import pandas as pd
from datetime import datetime
from collections import defaultdict
sys.path.append("..")

from model_utils.model import DeepSpeech2Model
from data_utils.dataloader import SpecgramGenerator, padding_batch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def tunning(lr, lr_key, num_passes=40, num_iterations_print=400, augmentation_config=augmentation_config):
    train_dataset = SpecgramGenerator(manifest="./iconect/models/pt_only_all_utt/train.csv",
                                      vocab_filepath="../models/baidu_en8k/vocab.txt",
                                      mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                      augmentation_config=augmentation_config,
                                      segmented=False)

    val_dataset = SpecgramGenerator(manifest="./iconect/models/pt_only_all_utt/val.csv",
                                    vocab_filepath="../models/baidu_en8k/vocab.txt",
                                    mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                    segmented=False)

    train_dataloader = DataLoader(train_dataset, batch_size=16,
                                  shuffle=True, num_workers=1,
                                  collate_fn=padding_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=16,
                                  shuffle=True, num_workers=1,
                                  collate_fn=padding_batch)
    vocab_list = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ds2_model = DeepSpeech2Model(
        vocab_size=28,
        num_conv_layers=2,
        num_rnn_layers=3,
        rnn_layer_size=1024,
        use_gru=True,
        pretrained_model_path="TBD",
        use_gpu=True)


    ds2_model.init_ext_scorer(1.4, 0.35, "../models/lm/common_crawl_00.prune01111.trie.klm", vocab_list=vocab_list)


    filename = datetime.now().strftime("gridsearch_%y%m%d-%H:%M:%S-lr{}-{}".format(lr, "_".join(lr_key)))
    output_dir = "./iconect/models/pt_only_all_utt/{}".format(filename)
    tensorboard_writer = SummaryWriter(log_dir="exp_tensorboard/{}".format(filename))
    ds2_model.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                    lr_key=lr_key,
                    learning_rate=lr,
                    gradient_clipping=400,
                    num_passes=num_passes,
                    num_iterations_print=num_iterations_print,
                    writer=tensorboard_writer,
                    output_dir=output_dir)

    outputs = defaultdict(list)
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
    df.to_pickle("./iconect/models/pt_only_all_utt/deepspeech_val_{}.pkl".format(filename))


if __name__ == "__main__":
    from itertools import product
    lr_keys = [["bias"], ["bias", "bn"], ["bn"]]
    lrs = [i*1e-6 for i in range(1, 21, 4)]
    for lr, lr_key in product(lrs, lr_keys):
        tunning(lr=lr, lr_key=lr_key, num_passes=40, num_iterations_print=400)

