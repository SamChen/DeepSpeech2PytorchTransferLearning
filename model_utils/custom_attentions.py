import os
import streamlit as st
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from typing import List, Tuple

#################################################################################
#
#                          Local self attention
#
#################################################################################
class pt_local_self_attention_layer(nn.MultiheadAttention):
    # it is a wrapper of multiheadattention with a special key_mask
    @staticmethod
    def attention_mask(seq_lengths, seq_length_max, query_index, window_size, device):
        assert query_index > -1, "query_index should not be negative!!!"

        batch_size = seq_lengths.size(0)
        atten_mask = torch.zeros((batch_size, seq_length_max), dtype=torch.bool)

        for batch_index, max_length in enumerate(seq_lengths):
            left_boundry = query_index - window_size
            left_boundry = left_boundry if left_boundry >-1 else 0
            right_boundry = query_index+window_size+1
            right_boundry = right_boundry if right_boundry < max_length else max_length
            atten_mask[batch_index, left_boundry: right_boundry] = True

        # atten_mask = atten_mask * padding_mask
        atten_mask = ~atten_mask
        return atten_mask.to(device)

class Local_SelfAttenion_Layer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, dropout=0.1, device="cpu"):
        super(Local_SelfAttenion_Layer, self).__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, input_dim)
        self.atten = pt_local_self_attention_layer(embed_dim=embed_dim, num_heads=num_heads, bias=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.device = device

    def loading_weights(self):
        pass

    def forward(self, x: PackedSequence, seq_lengths: Tensor, window_size: int) -> Tuple[PackedSequence, Tensor]:
        assert isinstance(x, PackedSequence)
        batch_first=False
        # pre-process
        x, x_lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=batch_first, padding_value=0.0)
        assert torch.equal(x_lengths,
                           seq_lengths.type_as(x_lengths)),"x_lengths: {}\nseq_lengths: {}".format(x_lengths, seq_lengths)

        x = self.in_proj(x)

        x_reconstructed, att_weights = self._attention(querys=x, keys=x, values=x,
                                          seq_lengths=x_lengths,
                                          window_size=window_size)

        x = x + self.dropout1(x_reconstructed)

        # post-process
        x = self.norm1(x)
        x = self.out_proj(x)

        results_packed = nn.utils.rnn.pack_padded_sequence(x,
                                                           lengths=x_lengths,
                                                           batch_first=batch_first)
        return results_packed, att_weights

    def _attention(self, querys, keys, values,
                   seq_lengths,
                   window_size):
        results = []
        att_weights_pack = []
        seq_length_max = seq_lengths[0]
        for query_index in range(seq_length_max):
            mask_keys = self.atten.attention_mask(seq_lengths=seq_lengths,
                                                  seq_length_max= seq_length_max,
                                                  query_index=query_index,
                                                  window_size=window_size,
                                                  device=self.device)

            # slice the query while keep the original dimensionality
            query = querys[query_index: query_index+1, :, :]

            new_frame, att_weights = self.atten(query, keys, values,
                                                key_padding_mask=mask_keys,
                                                need_weights=True)

            results.append(new_frame)
            att_weights_pack.append(att_weights)

        results_orig = torch.cat(results, dim=0)
        results_orig[results_orig!=results_orig] = 0

        # be careful! The shape of output of Pytorch's multihead_attention are slightly different between output and output weights.
        return results_orig, torch.cat(att_weights_pack, dim=1).contiguous().transpose(1,0)


#################################################################################
#
#                          attention
#
#################################################################################





class BaseAtten_Layer(nn.Module):
    def __init__(self, embed_dim, window_size,
                 key_dim=None, value_dim = None,
                 bias=False, attn_dropout=0.1,
                 temperature=1):
        super(BaseAtten_Layer, self).__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim if key_dim else self.embed_dim
        self.value_dim = value_dim if value_dim else self.embed_dim
        self.window_size = window_size
        self.temperature = temperature
        self.attn_dropout = attn_dropout

        self.wq = None
        self.wk = None
        self.wv = None
        self.post_score = None

    def forward(self, query, key, value, attn_mask=None):
        assert key.size() == value.size()
        assert key.size(0) == query.size(0)
        assert key.size(2) == query.size(2)
        assert (query.size(1) == 1) or (query.size(1) == key.size(1))

        q = self.wq(query) # Bx1xD -> Bx1xD  OR  BxTxD -> BxTxD.  One query OR All queries
        k = self.wk(key)   # BxTxD -> BxTxD
        v = self.wv(value) # BxTxD -> BxTxD

        scores = self.scoring(q, k, attn_mask)
        assert scores.size(0) == v.size(0)
        assert scores.size(2) == v.size(1)
        output = torch.matmul(scores, v)

        return output, scores.transpose(2, 1) # scores: Bx1xT -> BxTx1

    def scoring(self, query, keys, att_mask=None):
        pass

    @staticmethod
    def attention_mask(seq_lengths, seq_length_max, window_size,
                       query_index=None, device="cpu"):

        assert query_index > -1, "query_index should not be negative!!!"
        assert len(seq_lengths.size()) == 1, \
            "seq_lengths should be an 1D tensor, i.e. [len1, len2, ... , lenBatch_size]"

        batch_size = seq_lengths.size(0)
        atten_mask = torch.zeros((batch_size, 1, seq_length_max), dtype=torch.bool)

        stop_edge = None
        for batch_index, max_length in enumerate(seq_lengths):
            if query_index < max_length:
                left_boundry = query_index - window_size
                left_boundry = left_boundry if left_boundry >-1 else 0
                right_boundry = query_index + window_size+1
                right_boundry = right_boundry if right_boundry < max_length else max_length
                atten_mask[batch_index, 0, left_boundry: right_boundry] = True
            else:
                stop_edge = batch_index
                break

        atten_mask = ~atten_mask
        return atten_mask.to(device), stop_edge


class NTMAtten_Layer(BaseAtten_Layer):
    def __init__(self, embed_dim, window_size,
                 key_dim=None, value_dim = None,
                 bias=False, attn_dropout=0.1,
                 temperature=1):
        super(NTMAtten_Layer, self).__init__(embed_dim, window_size,
                                             key_dim, value_dim,
                                             bias, attn_dropout,
                                             temperature)

        self.wq = nn.Sequential(nn.Linear(self.key_dim,   self.embed_dim, bias=bias),
                                nn.Tanh())
        self.wk = nn.Sequential(nn.Linear(self.key_dim,   self.embed_dim, bias=bias),
                                nn.Tanh())
        self.wv = nn.Sequential(nn.Linear(self.value_dim, self.embed_dim, bias=bias),
                                nn.Tanh())
        self.post_score = nn.Sequential(nn.Tanh(),
                                        nn.Linear(self.embed_dim, 1, bias=False))

    def scoring(self, query, keys, att_mask=None):
        assert query.size()[2] == keys.size()[2]
        # only content base
        scores = self.post_score(query + keys) # BxTxD -> BxTx1
        scores = scores / self.temperature
        scores = scores.transpose(1,2) #BxTx1 -> Bx1xT

        if att_mask is not None:
            scores = scores.masked_fill_(att_mask, float("-inf"))
        return nn.functional.softmax(scores, dim=-1)



class DotAtten_Layer(BaseAtten_Layer):
    def __init__(self, embed_dim, window_size,
                 key_dim=None, value_dim = None,
                 bias=False, attn_dropout=0.1,
                 temperature=1):
        super(DotAtten_Layer, self).__init__(embed_dim, window_size,
                                             key_dim, value_dim,
                                             bias, attn_dropout,
                                             temperature)

        self.wq = nn.Linear(self.key_dim,   self.embed_dim, bias=bias)
        self.wk = nn.Linear(self.key_dim,   self.embed_dim, bias=bias)
        self.wv = nn.Linear(self.value_dim, self.embed_dim, bias=bias)
        self.dropout = nn.Dropout(self.attn_dropout)

    def scoring(self, query, keys, att_mask=None):
        assert query.size()[2] == keys.size()[2]
        query *= self.embed_dim ** -0.5
        scores = torch.bmm(query, keys.transpose(1, 2))

        if att_mask is not None:
            scores = scores.masked_fill_(att_mask, float("-inf"))
        scores = nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return scores

    def reconstruct(self, scores, values):
        assert scores.size(0) == values.size(0)
        return torch.matmul(scores, values)


