#!/usr/bin/env python

import copy
import numpy as np
import unittest
import torch
import torch.nn as nn
import sys
sys.path.append("../model_utils/")

from basic_ops import iter_packedsequence, packedsequences_merge


class packedsequence_tests(unittest.TestCase):
    def iter_packedsequence(self):
        num_layers = 10
        # batch_size=5, seq_length=6, feature_size=3
        data = [torch.rand(5, 6, 3).requires_grad_()  for i in range(num_layers)]

        packed_data_list = []
        for i in data:
            temp, batchs, *_ = pack_sequence(i)
            packed_data_list.append(temp)

        packed_data, data = packedsequences_merge(packed_data_list)
        # check the interator works properately.
        for frame, t in interatie_packedsequences(packed_data):
            for i in range(num_layers):
                assert torch.equal(frame[:, i,:],data[i][:, t, :]), "{}\n{} \n==============\n {}".format(i,frame[:, i,:],data[i][:, t, :])

    def packedsequences_merge():
        pass


if __name__ == "__main__":
    unittest.main()

