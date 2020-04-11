#!/usr/bin/env python
import os
import sys
import streamlit as st
import torch
import torch.nn as nn
sys.path.append("../model_utils")
from custom_attentions import Local_SelfAttenion_Layer

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import List, Tuple


if __name__ == "__main__":
    st.caching.clear_cache()
    st.title("Local attention prototype")
    input_dim = 20
    x   = torch.rand(8, 3, input_dim, dtype=torch.float)

    pad_mask = torch.ones_like(x, dtype=torch.bool)
    seq_lengths = torch.tensor([8, 7, 6])
    for index, length in enumerate(seq_lengths):
        pad_mask[length:, index, :] = False

    window_size = 3
    x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=False)

    test_nn = Local_SelfAttenion_Layer(embed_dim=8, num_heads=1, input_dim=input_dim)
    results, att_weights = test_nn(x, seq_lengths, window_size=2)
    st.write(results[0].size())
    st.write(att_weights.sum(dim=-1, keepdim=True))

