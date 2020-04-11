import copy
import numpy as np
import unittest
import torch
import torch.nn as nn
import sys
sys.path.append("../model_utils/")
import custom_attentions
"""
This unittest is for testing the attention mechanism.
Including, generate mask for local attention, attention mechanism's forward backward operation.

More tests will be added as more attention mechanisms are implemented.
"""


class LocalSelfAttenMask_test(unittest.TestCase):
    def test_mask_regular(self):
        seq_length = 10
        query_index = 5
        window_size = 2
        test_sample = torch.ones((1, seq_length), dtype=torch.bool)
        # all position that needs to be masked will be set to True
        test_sample[:, query_index -
                    window_size: query_index+window_size+1] = False
        embed_dim = 1
        num_heads = 1

        test_atten = custom_attentions.pt_local_self_attention_layer(
            embed_dim=embed_dim, num_heads=num_heads, bias=False)
        test_mask = test_atten.attention_mask(torch.tensor([seq_length]), seq_length_max=seq_length,
                                              query_index=query_index, window_size=window_size, device="cpu")

        assert torch.equal(test_sample, test_mask), "test_sample: {}, test_mask: {}".format(
            test_sample, test_mask)

    def test_mask_edgecase_leftmost(self):
        seq_length = 10
        query_index = 0
        window_size = 2
        test_sample = torch.zeros((1, seq_length), dtype=torch.bool)
        # all position that needs to be masked will be set to True
        test_sample[:, 3:] = True
        embed_dim = 1
        num_heads = 1

        test_atten = custom_attentions.pt_local_self_attention_layer(
            embed_dim=embed_dim, num_heads=num_heads, bias=False)
        test_mask = test_atten.attention_mask(torch.tensor([seq_length]), seq_length_max=seq_length,
                                              query_index=query_index, window_size=window_size, device="cpu")

        assert torch.equal(test_sample, test_mask), "test_sample: {}, test_mask: {}".format(
            test_sample, test_mask)

    def test_mask_edgecase_rightmost(self):
        seq_length = 10
        query_index = 0
        window_size = 2
        test_sample = torch.zeros((1, seq_length), dtype=torch.bool)
        # all position that needs to be masked will be set to True
        test_sample[:, :seq_length-window_size] = True
        embed_dim = 1
        num_heads = 1

        test_atten = custom_attentions.pt_local_self_attention_layer(
            embed_dim=embed_dim, num_heads=num_heads, bias=False)
        test_mask = test_atten.attention_mask(torch.tensor([seq_length]), seq_length_max=seq_length,
                                              query_index=seq_length, window_size=window_size, device="cpu")

        assert torch.equal(test_sample, test_mask), "test_sample: {}, test_mask: {}".format(
            test_sample, test_mask)


class LocalSelfAtten_mask_compute_test(unittest.TestCase):
    def test_forward(self):
        embed_dim = 6
        num_heads = 1
        seq_lengths = torch.tensor([10, 10, 10])
        input_padded = torch.rand((20, 3, 6))
        input_padded[10:] = 0.0
        input_nopad = copy.deepcopy(input_padded[:10, :, :])

        test_atten_padded = custom_attentions.pt_local_self_attention_layer(
            embed_dim=embed_dim, num_heads=num_heads, bias=False)
        test_atten_nopad = copy.deepcopy(test_atten_padded)

        output_nopad, weights_nopad = test_atten_nopad(query=input_nopad,
                                                       key=input_nopad,
                                                       value=input_nopad,
                                                       key_padding_mask=None,
                                                       need_weights=True,
                                                       attn_mask=None)
        output_padded = []
        weights_padded = []
        # we assume that we already know the maximum valide length which is 10 in here.
        # if we don't know the maximum valid length. This code will cause problem.
        for i in range(10):
            mask = test_atten_padded.attention_mask(seq_lengths, seq_length_max=20,
                                                    query_index=i, window_size=10,
                                                    device="cpu")

            query = input_padded[i: i+1, :, :]
            temp, weights = test_atten_padded(query=query,
                                              key=input_padded,
                                              value=input_padded,
                                              key_padding_mask=mask,
                                              need_weights=True,
                                              attn_mask=None)
            output_padded.append(temp)
            weights_padded.append(weights)
        output_padded = torch.cat(output_padded, dim=0)
        weights_padded = torch.cat(weights_padded, dim=1)


        # The outputs are not exactly the same. Based on the answer from https://discuss.pytorch.org/t/torch-bmm-s-performance-is-a-little-bit-strange-to-me/70239, this should be normal. It is unlikely cased by my code.
        #
        # check unmasked part. They should be similar
        assert (weights_nopad - weights_padded[:, :10, :10]).abs().max() < 1e-7
        assert (output_padded[:10, :, :] - output_nopad).abs().max(
        ) < 1e-7
        # check masked part. All masked part should be zero
        assert torch.equal(output_padded[10:,:,:], torch.zeros_like(output_padded[10:,:,:])),output_padded[10:]

    def test_backward(self):
        pass


class DotAtten_Mask_test(unittest.TestCase):
    def test_mask_regular(self):
        seq_length = 10
        query_index = 5
        window_size = 2
        test_sample = torch.ones((1, 1, seq_length), dtype=torch.bool)
        # all position that needs to be masked will be set to True
        test_sample[:, :,query_index -
                    window_size: query_index+window_size+1] = False
        embed_dim = 1
        output_dim = 1

        test_atten = custom_attentions.DotAtten_layer(
            embed_dim, output_dim,
            key_dim=None, value_dim = None,
            bias=False)

        test_mask = test_atten.attention_mask(
            torch.tensor([seq_length]),
            seq_length_max=seq_length,
            query_index=query_index,
            window_size=window_size,
            device="cpu")

        assert torch.equal(test_sample, test_mask), "test_sample: {}, test_mask: {}".format(
            test_sample, test_mask)

    def test_mask_edgecase_leftmost(self):
        seq_length = 10
        query_index = 0
        window_size = 2
        test_sample = torch.zeros((1, 1, seq_length), dtype=torch.bool)
        # all position that needs to be masked will be set to True
        test_sample[:, :, 3:] = True
        embed_dim = 1
        output_dim = 1

        test_atten = custom_attentions.DotAtten_layer(
            embed_dim, output_dim,
            key_dim=None, value_dim = None,
            bias=False)
        test_mask = test_atten.attention_mask(
            torch.tensor([seq_length]),
            seq_length_max=seq_length,
            query_index=query_index,
            window_size=window_size, device="cpu")

        assert torch.equal(test_sample, test_mask), "test_sample: {}, test_mask: {}".format(
            test_sample, test_mask)

    def test_mask_edgecase_rightmost(self):
        seq_length = 10
        query_index = 0
        window_size = 2
        test_sample = torch.zeros((1, 1, seq_length), dtype=torch.bool)
        # all position that needs to be masked will be set to True
        test_sample[:, :, :seq_length-window_size] = True
        embed_dim = 1
        output_dim = 1

        test_atten = custom_attentions.DotAtten_layer(
            embed_dim, output_dim,
            key_dim=None, value_dim = None,
            bias=False)

        test_mask = test_atten.attention_mask(
            torch.tensor([seq_length]),
            seq_length_max=seq_length,
            query_index=seq_length,
            window_size=window_size, device="cpu")

        assert torch.equal(test_sample, test_mask), "test_sample: {}, test_mask: {}".format(test_sample, test_mask)


class LocalDotAttn_mask_compute_test(unittest.TestCase):
    def test_forward(self):
        embed_dim = 6
        output_dim = 6
        seq_lengths = torch.tensor([10, 10, 10])
        input_padded = torch.rand((3, 20, 6))
        input_padded[10:] = 0.0
        input_nopad = copy.deepcopy(input_padded[:, :10, :])

        test_atten_padded = custom_attentions.DotAtten_layer(
            embed_dim, output_dim,
            key_dim=None, value_dim = None,
            bias=False)
        test_atten_nopad = copy.deepcopy(test_atten_padded)

        output_nopad, weights_nopad = test_atten_nopad(
            query=input_nopad,
            key=input_nopad,
            value=input_nopad,
            attn_mask=None)

        output_padded = []
        weights_padded = []
        # we assume that we already know the maximum valide length which is 10 in here.
        # if we don't know the maximum valid length. This code will cause problem.
        for i in range(10):
            mask = test_atten_padded.attention_mask(
                seq_lengths, seq_length_max=20,
                query_index=i, window_size=10,
                device="cpu")
            # We only tend to test the calculation. Making sure the masked part will not be involved in the computation. Test mask generation is another unittest.

            query = input_padded[:, i: i+1, :]
            temp, weights = test_atten_padded(query=query,
                                              key=input_padded,
                                              value=input_padded,
                                              attn_mask=mask)
            output_padded.append(temp)
            weights_padded.append(weights)

        output_padded = torch.cat(output_padded, dim=1)
        weights_padded = torch.cat(weights_padded, dim=1)


        assert (weights_nopad - weights_padded[:, :10, :10]).abs().max() < 1e-7
        assert (output_padded[:10, :, :] - output_nopad).abs().max(
        ) < 1e-7
        # check masked part. All masked part should be zero
        assert torch.equal(output_padded[10:,:,:], torch.zeros_like(output_padded[10:,:,:])),output_padded[10:]

    def test_backward(self):
        pass


if __name__ == "__main__":
    unittest.main()
