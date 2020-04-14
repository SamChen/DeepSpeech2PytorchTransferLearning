#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


from .custom_layers import Conv_bn_mask
from .custom_gru import DynamicGRUlayer, GRU_hiddenCell
from .custom_attentions import Local_SelfAttenion_Layer, DotAtten_Layer
from .paddle_weights_loading import load_weights_bottleneck, load_weights_hiddenlayers


class deepspeech_hiddenLayers(nn.Module):
    def __init__(self,device="cuda"):
        self.device = device
        super(deepspeech_hiddenLayers, self).__init__()
        self.conv_bn_mask0 = Conv_bn_mask(ichannel=1,
                                          ochannel=32,
                                          kernel_size=(11, 41),
                                          stride=(3, 2),
                                          padding=(5, 20),
                                          bias=False,
                                          track_running_stats=True)

        self.conv_bn_mask1 = Conv_bn_mask(ichannel=32,
                                          ochannel=32,
                                          kernel_size=(11, 21),
                                          stride=(1, 2),
                                          padding=(5, 10),
                                          bias=False,
                                          track_running_stats=True)

        self.bigru0 = DynamicGRUlayer(GRU_hiddenCell, input_size=41 * 32, hidden_size=1024, gate_act="sigmoid", state_act="relu")
        self.bigru1 = DynamicGRUlayer(GRU_hiddenCell, input_size=2048, hidden_size=1024,gate_act="sigmoid", state_act="relu")
        self.bigru2 = DynamicGRUlayer(GRU_hiddenCell, input_size=2048, hidden_size=1024,gate_act="sigmoid", state_act="relu")


    def load_paddle_pretrained(self, model_path):
        """
        the weights relationship is hardcoded.
        """
        pretrained_weights = load_weights_hiddenlayers(model_path=model_path)
        check_dict = self.state_dict()
        for key in check_dict:
            if 'num_batches_tracked' in key:
                continue
            assert key in pretrained_weights
            check_dict[key] = torch.from_numpy(pretrained_weights[key])
        self.load_state_dict(check_dict)


    def initial_hidden_states_generation(self, batch_size):
        return [torch.zeros((2, batch_size, 1024)).to(self.device), ]

    def forward(self, input):
        seq_masks  = input["cnn_masks"]
        x = input["specgrams"].to(self.device)

        batch_size = x.shape[0]
        init_state0 = self.initial_hidden_states_generation(batch_size)
        init_state1 = self.initial_hidden_states_generation(batch_size)
        init_state2 = self.initial_hidden_states_generation(batch_size)

        assert (x != x).sum() == 0, "{}, value of x: {}".format( input["uttid"],x)
        x = self.conv_bn_mask0(x, seq_masks[0])
        assert (x != x).sum() == 0, "{}, value of x: {}, num of not nan: {}, conv_weights: {}".format( input["uttid"],x, (~(x != x)).sum(), self.conv_bn_mask0.conv.weight)
        x = self.conv_bn_mask1(x, seq_masks[1])

        x = x.transpose(2, 1).contiguous()

        flattened_x = x.reshape(batch_size, -1, 41 * 32)
        assert (flattened_x != flattened_x).sum() == 0, "{}, value of x: {}".format( input["uttid"],flattened_x)

        flattened_x = nn.utils.rnn.pack_padded_sequence(flattened_x, seq_masks[1].flatten(), batch_first=True)

        flattened_x, _ = self.bigru0(flattened_x, init_state0)
        assert (flattened_x[0] != flattened_x[0]).sum() == 0, "value of x: {}".format(flattened_x[0])

        flattened_x, _ = self.bigru1(flattened_x, init_state1)
        assert (flattened_x[0] != flattened_x[0]).sum() == 0, "value of x: {}".format(flattened_x[0])

        flattened_x, _ = self.bigru2(flattened_x, init_state2)
        assert (flattened_x[0] != flattened_x[0]).sum() == 0, "value of x: {}".format(flattened_x[0])

        output = flattened_x
        # the masks contains the length information of each sample in the batch
        sample_lengths = seq_masks[1]
        return output, sample_lengths

class deepspeech_LocalDotAttenLayers(nn.Module):
    def __init__(self, embed_dim,
                 init_strategy, window_size,
                 input_dim=2048, output_dim=29,
                 bias=False,device="cuda"):
        super(deepspeech_LocalDotAttenLayers, self).__init__()
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy
        self.device = device
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_ratio = 0.3

        self.query_proj = nn.Linear(output_dim, embed_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.atten = DotAtten_Layer(embed_dim=embed_dim,
                                    window_size=window_size,
                                    bias=bias, attn_dropout=0.2)
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def generate_query0(self, batch_size, output_dim):
        if self.init_strategy == "random":
            # query0 = torch.rand([batch_size, 1, output_dim])
            query0 = torch.ones([batch_size, 1, output_dim])
        elif self.init_strategy == "fix":
            query0 = torch.ones([batch_size, 1, output_dim])
            query0[:,:,-1] = 9
        return query0.to(self.device)

    def load_paddle_pretrained(self):
        pass

    def forward(self, x, sample_lengths, window_size):
        assert isinstance(x, PackedSequence)
        x, x_lengths = pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        assert torch.equal(x_lengths, sample_lengths.type_as(x_lengths))

        # x = self.input_norm(x)
        x = self.input_proj(x)
        query = self.generate_query0(x.size(0), self.output_dim)

        outputs = []
        outputs_weights = []
        batchsize = len(sample_lengths)

        for query_index in range(sample_lengths[0]):
            query = self.query_proj(query)
            mask, stop_edge = self.atten.attention_mask(
                seq_lengths=sample_lengths,
                seq_length_max=sample_lengths[0],
                query_index=query_index,
                window_size=window_size,
                device=self.device)

            if stop_edge is not None:
                x = torch.narrow(x, dim=0, start=0, length=stop_edge)
                query = torch.narrow(query, dim=0, start=0, length=stop_edge)
                mask = torch.narrow(mask, dim=0, start=0, length=stop_edge)
                blank_out = torch.zeros(batchsize-stop_edge, 1, self.embed_dim).to(self.device)
                blank_wgts = torch.zeros(batchsize-stop_edge, *out_wgts.shape[1:]).to(self.device)

            out, out_wgts = self.atten(query, key=x, value=x, attn_mask=mask)

            if stop_edge is not None:
                out = torch.cat([out, blank_out], dim=0)
                out_wgts = torch.cat([out_wgts, blank_wgts], dim=0)

            outputs_weights.append(out_wgts)
            # output of timestep t will be the query of t+1
            out = self.dropout(out)
            query = self.output_proj(out)
            outputs.append(query)

        outputs_weights = torch.cat(outputs_weights, dim=1)
        outputs_weights = pack_padded_sequence(outputs_weights, sample_lengths,
                                               batch_first=True)
        outputs = torch.cat(outputs, dim=1)
        outputs = pack_padded_sequence(outputs, sample_lengths,
                                       batch_first=True)

        return outputs, sample_lengths, outputs_weights



class deepspeech_bottleneckLayer(nn.Module):
    def __init__(self, device="cuda"):
        super(deepspeech_bottleneckLayer, self).__init__()
        self.device = device
        self.vocab_size=28
        # 28 of char + 1 blank
        # vocab list:  ["'", ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.bottleneck = nn.Linear(2048, self.vocab_size + 1)

    def load_paddle_pretrained(self, model_path):
        pretrained_weights = load_weights_bottleneck(model_path=model_path)
        check_dict = self.state_dict()
        for key in check_dict:
            if 'num_batches_tracked' in key:
                continue
            assert key in pretrained_weights
            check_dict[key] = torch.from_numpy(pretrained_weights[key])
        self.load_state_dict(check_dict)

    def forward(self, x, sample_lengths):
        assert isinstance(x, PackedSequence)
        data_x, batch_sizes, sorted_indices, unsorted_indices = x
        data_x = self.bottleneck(data_x)
        output = nn.utils.rnn.PackedSequence(data_x, batch_sizes, sorted_indices, unsorted_indices)
        return output, sample_lengths


class deepspeech_LocalSelfAttenLayer(nn.Module):
    def __init__(self, device="cuda"):
        super(deepspeech_LocalSelfAttenLayer, self).__init__()
        self.device = device
        self.atten = Local_SelfAttenion_Layer(input_dim=2048, embed_dim=512, num_heads=1,device=device)

    def load_paddle_pretrained(self):
        pass

    def forward(self, x: PackedSequence, sample_lengths, window_size: int):
        assert isinstance(x, PackedSequence)
        output, att_weights = self.atten(x, sample_lengths, window_size)
        return output, sample_lengths, att_weights


class deepspeech_outputLayer(nn.Module):
    def __init__(self, device="cuda"):
        super(deepspeech_outputLayer, self).__init__()
        self.device = device

    def load_paddle_pretrained(self, model_path):
        pass

    def forward(self, x, sample_lengths):
        data_x, batch_sizes, sorted_indices, unsorted_indices = x
        if self.training is True:
            data_x = data_x.log_softmax(-1)
            output = nn.utils.rnn.PackedSequence(data_x, batch_sizes, sorted_indices, unsorted_indices)

            # this is a special request for pytorch CTCloss. Batchsize must be the second dimension
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        else:

            # In paddlepaddle DeepSpeech, they use softmax instead of logsoftmax!!!
            # for their CTCLoss
            # Since I did not modify the beam search part. Currently, I intend to keep it this way.
            data_x = data_x.softmax(-1)
            output = nn.utils.rnn.PackedSequence(data_x, batch_sizes, sorted_indices, unsorted_indices)
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, sample_lengths

class deepspeech_orig(nn.Module):
    def __init__(self,device="cuda"):
        super(deepspeech_orig, self).__init__()
        self.device = device
        self.deepspeech_hiddenlayers = deepspeech_hiddenLayers(self.device)
        self.deepspeech_bottleneck   = deepspeech_bottleneckLayer(self.device)
        self.deepspeech_output       = deepspeech_outputLayer(self.device)

    def load_paddle_pretrained(self, model_path):
        self.deepspeech_hiddenlayers.load_paddle_pretrained(model_path)
        self.deepspeech_bottleneck.load_paddle_pretrained(model_path)
        self.deepspeech_output.load_paddle_pretrained(model_path)

    def forward(self, input):
        x = self.deepspeech_hiddenlayers(input)
        x = self.deepspeech_bottleneck(*x)
        x, lengths = self.deepspeech_output(*x)
        # x contains two variables
        return x, lengths, []

class deepspeech_newbottleneck(nn.Module):
    def __init__(self,device="cuda"):
        super(deepspeech_newbottleneck, self).__init__()
        self.device = device
        self.deepspeech_hiddenlayers = deepspeech_hiddenLayers(self.device)
        self.deepspeech_bottleneck   = deepspeech_bottleneckLayer(self.device)
        self.deepspeech_output       = deepspeech_outputLayer(self.device)

    def load_paddle_pretrained(self, model_path):
        self.deepspeech_hiddenlayers.load_paddle_pretrained(model_path)
        self.deepspeech_output.load_paddle_pretrained(model_path)

    def forward(self, input):
        x = self.deepspeech_hiddenlayers(input)
        x = self.deepspeech_bottleneck(*x)
        x, lengths = self.deepspeech_output(*x)
        # x contains two variables
        return x, lengths, []


class deepspeech_LocalSelfAtten(nn.Module):
    def __init__(self,device="cuda"):
        super(deepspeech_LocalSelfAtten, self).__init__()
        self.device = device
        self.deepspeech_hiddenlayers = deepspeech_hiddenLayers(self.device)
        self.deepspeech_exp_layers = deepspeech_LocalSelfAttenLayer(self.device)
        self.deepspeech_bottleneck = deepspeech_bottleneckLayer(self.device)
        self.deepspeech_output = deepspeech_outputLayer(self.device)

    def load_paddle_pretrained(self, model_path):
        self.deepspeech_hiddenlayers.load_paddle_pretrained(model_path)
        self.deepspeech_bottleneck.load_paddle_pretrained(model_path)
        self.deepspeech_output.load_paddle_pretrained()

    def forward(self, input):
        x,lengths = self.deepspeech_hiddenlayers(input)
        x, lengths, att_weights = self.deepspeech_exp_layers(x, lengths, window_size=5)
        x, lengths = self.deepspeech_bottleneck(x, lengths)
        x, lengths = self.deepspeech_output(x, lengths)
        return x, lengths, []

class deepspeech_LocalDotAtten(nn.Module):
    def __init__(self,device="cuda"):
        super(deepspeech_LocalDotAtten, self).__init__()
        self.device = device
        self.deepspeech_hiddenlayers = deepspeech_hiddenLayers(self.device)
        self.deepspeech_exp_layers = deepspeech_LocalDotAttenLayers(embed_dim=768,
                                                                    init_strategy="fix",
                                                                    window_size=5,
                                                                    device=self.device)
        self.deepspeech_output = deepspeech_outputLayer(self.device)

        self.window_size = 5# window_size
    def load_paddle_pretrained(self, model_path):
        self.deepspeech_hiddenlayers.load_paddle_pretrained(model_path)
        self.deepspeech_output.load_paddle_pretrained()

    def forward(self, input):
        x,lengths = self.deepspeech_hiddenlayers(input)
        x, lengths, att_weights = self.deepspeech_exp_layers(x, lengths, self.window_size)
        x, lengths = self.deepspeech_output(x, lengths)
        return x, lengths, [att_weights, ]

