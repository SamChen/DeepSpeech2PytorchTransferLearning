import sys
import os
import pandas as pd
from collections import defaultdict
sys.path.append("..")

from model_utils.model import DeepSpeech2Model
from model_utils.network import deepspeech_LocalSelfAtten, deepspeech_orig, deepspeech_LocalDotAtten
from data_utils.dataloader import SpecgramGenerator#, padding_batch
from torch.utils.data import DataLoader




def infer(source_dir, target_dir, filename):
    vocab_list = ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ds2_model = DeepSpeech2Model(network=deepspeech_orig,
                                 vocab_size=28,
                                 pretrained_model_path="TBD",
                                 device="cuda")

    ds2_model.init_ext_scorer(1.4, 0.35, "../models/lm/common_crawl_00.prune01111.trie.klm", vocab_list=vocab_list)

    ###############################
    ## load data
    ###############################
    test_dataset = SpecgramGenerator(manifest=os.path.join(source_dir, filename),
                                      vocab_filepath="../models/baidu_en8k/vocab.txt",
                                      mean_std_filepath="../models/baidu_en8k/mean_std.npz",
                                      max_duration=float('inf'),
                                      min_duration=0.0,
                                      segmented=False)

    dataloader = DataLoader(test_dataset, batch_size=16,
                    shuffle=False, num_workers=4,
                   collate_fn=SpecgramGenerator.padding_batch)


    ###############################
    ## infering
    ###############################
    outputs = defaultdict(list)
    for i_batch, sample_batched in enumerate(dataloader):
        try:
            batch_results = ds2_model.infer_batch_probs(infer_data=sample_batched)
            batch_transcripts_beam = ds2_model.decode_batch_beam_search(probs_split=batch_results,
                                                                        beam_alpha=3, #1.4,
                                                                        beam_beta=0.35,
                                                                        beam_size=500,
                                                                        cutoff_prob=1.0,
                                                                        cutoff_top_n=40,
                                                                        vocab_list=vocab_list,
                                                                        num_processes=5)
            outputs["uttid"].extend(sample_batched["uttid"])
            outputs["status"].extend(["success"]*len(sample_batched["uttid"]))
            outputs["asr"].extend(batch_transcripts_beam)
        except MemoryError:
            outputs["uttid"].extend(sample_batched["uttid"])
            outputs["status"].extend(["out_of_memory"]*len(sample_batched["uttid"]))
            outputs["asr"].extend([""]*len(sample_batched["uttid"]))

        outputs["text"].extend(sample_batched["trans"])

    ###############################
    ## saving
    ###############################
    ### filename_prefix = os.path.split(filename)[0]
    df = pd.DataFrame.from_dict(outputs)
    df.to_csv(os.path.join(target_dir, filename), index=False)


if __name__ == "__main__":
    filename = sys.argv[1]

    infer("./iconect/csvs_orig", "./iconect/transcripts2", filename)

