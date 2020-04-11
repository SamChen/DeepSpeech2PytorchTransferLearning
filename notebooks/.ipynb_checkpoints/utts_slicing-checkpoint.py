import gentle
import pandas as pd
from collections import defaultdict
import copy
from TEST_utt_split import utterances_split
from multiprocessing import Process, Pool
import itertools
import os
import re

def dataset_slicing(dataset, chunk_size, params):
    previous = 0
    for i in range(chunk_size, len(dataset), chunk_size):
        params["dataset"] = copy.deepcopy(dataset[previous:i])
        # deepcopy is necessory for avoid data sharing in different process
        yield copy.deepcopy(params)
        previous = i

    params["dataset"] = copy.deepcopy(dataset[previous:])
    # deepcopy is necessory for avoid data sharing in different process
    yield copy.deepcopy(params)

def worker(params:dict):
    splitting = utterances_split()
    all_utts = []
    dataset = params["dataset"]
    max_duration = params["max_duration"]
    max_gap = params["max_gap"]
    disfluencies = params["disfluencies"]

    for index, row in dataset.iterrows():
        new_utts = splitting.utt_split(row, max_duration=max_duration, max_gap=max_gap, disfluencies=disfluencies)
        all_utts.extend(new_utts)

    return all_utts

def multi_task(dataset, params, processes=3):
    pool = Pool(processes = processes)
    result = pool.map(worker, dataset_slicing(dataset, 100, params))
    pool.close()
    return list(itertools.chain(*result))

def main(root_dir, target_dir, params, subfix=""):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    files = ["train.csv", "test.csv", "val.csv"]
    for _,_,fs in os.walk(root_dir):
        assert files == fs, "{}".format(fs)
        # files.extend(fs)

    for file in files:
        if not re.match("\w+\.csv", file):
            continue

        filename = re.sub(r"\.csv", "_{}.csv".format(subfix), file)
        dataset = pd.read_csv(os.path.join(root_dir, file))

        result = multi_task(dataset, copy.deepcopy(params), processes=14)
        pd.DataFrame(result).to_csv(os.path.join(target_dir, filename), index=False)


if __name__== "__main__":
    root_dir = "iconect/models/pt_only_all_utt_test/data"
    target_dir = "iconect/models/pt_only_all_utt_test/data"

    params = {"max_duration":7.0,
              "max_gap": 1.0,
              "disfluencies": set(['uh', 'um'])
              }
    main(root_dir, target_dir, params, subfix="short")

    # params = {"max_duration":100000.0,
    #           "max_gap": 1.0,
    #           "disfluencies": set(['uh', 'um'])
    #           }
    # main(root_dir, target_dir, params, subfix="long")
#
#
#
    # root_dir = "iconect/models/pt_only_all_utt3/"
    # target_dir = "iconect/models/pt_only_all_utt_sliced3/"
#
    # params = {"max_duration":7.0,
    #           "max_gap": 1.0,
    #           "disfluencies": set(['uh', 'um'])
    #           }
    # main(root_dir, target_dir, params, subfix="short")
#
#
    # params = {"max_duration":100000.0,
    #           "max_gap": 1.0,
    #           "disfluencies": set(['uh', 'um'])
    #           }
    # main(root_dir, target_dir, params, subfix="long")
