import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
import numbers

'''
Some helper classes for writing custom TorchScript LSTMs.

Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.

A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
'''


def script_lstm(input_size, hidden_size, num_layers, bias=True,
                batch_first=False, dropout=False, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first

    if bidirectional:
        stack_type = StackedRNN2
        layer_type = BidirRNNLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LSTMCell, input_size, hidden_size],
                      other_layer_args=[LSTMCell, hidden_size * dirs,
                                        hidden_size])


def script_lnlstm(input_size, hidden_size, num_layers, bias=True,
                  batch_first=False, dropout=False, bidirectional=False,
                  decompose_layernorm=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LayerNormLSTMCell, input_size, hidden_size,
                                        decompose_layernorm],
                      other_layer_args=[LayerNormLSTMCell, hidden_size * dirs,
                                        hidden_size, decompose_layernorm])


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]

def torch_tile(tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,n,1).view(-1,tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0,1).repeat(1,1,n).view(tensor.shape[0], -1)
    
class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, gate_act="sigmoid", state_act="tanh"):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if gate_act == "sigmoid":
            self.gate_activation = torch.sigmoid
        elif gate_act == "relu":
            self.gate_activation = torch.relu

        if state_act == "tanh":
            self.state_activation = torch.tanh
        elif state_act == "relu":
            self.state_activation = torch.relu

        # order: w_u, w_r, w_c
        self.weight_i = Parameter(torch.randn( input_size , 3 * hidden_size))
        self.weight_h = Parameter(torch.randn( hidden_size, 3 * hidden_size))

        # order: b_u, b_r, b_c
        self.bias = Parameter(torch.randn(3 * hidden_size))
        self.bn = nn.BatchNorm1d(3 * hidden_size)

    # @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tensor) -> Tensor
        # check batch size matches
        hx = state
        assert input.shape[0] == state.shape[0]
        assert input.shape[1] == self.weight_i.shape[0]
        assert hx.shape[1] == self.weight_h.shape[0]

        gates = torch.mm(input, self.weight_i)
        gates = self.bn(gates) # deepspeech only normalize the input part
        gates += self.bias
        u,r,c = gates.chunk(3, 1)
        u_h,r_h,c_h = self.weight_h.chunk(3,1)


        u += torch.mm(hx, u_h.t())
        r += torch.mm(hx, r_h.t())
        u = self.gate_activation(u)
        r = self.gate_activation(r)
        
        c += torch.mm((r * hx), c_h.t())
        c = self.state_activation(c)

        hy = ( (1.0 - u) * hx) + (u * c)

        return hy


class RNNLayer(jit.ScriptModule):
    def __init__(self, cell, **cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(**cell_args)

    @jit.script_method
    def forward(self, input, hidden):
        # type: (PackedSequence, Tensor) -> Tuple[PackedSequence, Tensor]
        assert isinstance(input, PackedSequence)
        assert isinstance(hidden, torch.Tensor)

        input, batch_sizes, sorted_indices, unsorted_indices = input
        output = []
        input_offset = torch.tensor(0)
        last_batch_size = batch_sizes[0]
        finished_hiddens = []

        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                finished_hiddens.append(hidden[-dec:])
                hidden = hidden[:-dec]
            last_batch_size = batch_size

            hidden = self.cell(step_input, hidden)
            output.append(hidden)

        # append the finished hidden for the longest sequences in that batch
        finished_hiddens.append(hidden)
        # let the hiddens' order matches the input sequence order
        finished_hiddens.reverse()

        hidden = torch.cat(finished_hiddens, 0)
        assert hidden.size(0) == batch_sizes[0]

        output = torch.cat(output, 0)
        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)

        return output, hidden

class ReverseRNNLayer(jit.ScriptModule):
    def __init__(self, cell, **cell_args):
        super(ReverseRNNLayer, self).__init__()
        self.cell = cell(**cell_args)

    @jit.script_method
    def forward(self, input, hidden):
        # type: (PackedSequence, Tensor) -> Tuple[PackedSequence, Tensor]
        assert isinstance(input, PackedSequence)

        input, batch_sizes, sorted_indices, unsorted_indices = input

        output = []
        input_offset = torch.tensor(input.shape[0])
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden

        hidden = hidden[:last_batch_size]

        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                # add more initial hiddens for new sequences's last step
                hidden = torch.cat((hidden, initial_hidden[last_batch_size:batch_size]), 0)
            last_batch_size = batch_size

            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            hidden = self.cell(step_input, hidden)
            output.append(hidden)

        output.reverse()
        output = torch.cat(output, 0)
        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)

        return output, hidden


class BidirRNNLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, **cell_args):
        super(BidirRNNLayer, self).__init__()
        self.directions = nn.ModuleList([
            RNNLayer(cell, **cell_args),
            ReverseRNNLayer(cell, **cell_args),
        ])

    # @jit.script_method
    def forward(self, input, states):
        # type: (PackedSequence, List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]
        # List[RNNState]: [forward RNNState, backward RNNState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tensor], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        _,batch_size, sorted_indices, unsorted_indices = input
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out[0]]
            output_states += [out_state]
            i += 1
        outputs = nn.utils.rnn.PackedSequence(torch.cat(outputs, -1), batch_size, sorted_indices, unsorted_indices)
        # output_states = torch.cat(output_states, 0)
        return outputs, output_states


def init_stacked_rnn(num_layers, layer, first_layer_args, other_layer_args):
    # stack multiple RNN layers together
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedRNN(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedRNN, self).__init__()
        self.layers = init_stacked_rnn(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tensor]) -> Tuple[Tensor, List[Tensor]]
        # List[RNNState]: One state per layer
        output_states = jit.annotate(List[Tensor], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedRNN2(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedRNN2, self).__init__()
        self.layers = init_stacked_rnn(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[List[Tensor]]) -> Tuple[Tensor, List[List[Tensor]]]
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


# class StackedLSTMWithDropout(jit.ScriptModule):
#     # Necessary for iterating through self.layers and dropout support
#     __constants__ = ['layers', 'num_layers']
#
#     def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
#         super(StackedLSTMWithDropout, self).__init__()
#         self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
#                                         other_layer_args)
#         # Introduces a Dropout layer on the outputs of each LSTM layer except
#         # the last layer, with dropout probability = 0.4.
#         self.num_layers = num_layers
#
#         if (num_layers == 1):
#             warnings.warn("dropout lstm adds dropout layers after all but last "
#                           "recurrent layer, it expects num_layers greater than "
#                           "1, but got num_layers = 1")
#
#         self.dropout_layer = nn.Dropout(0.4)
#
#     @jit.script_method
#     def forward(self, input, states):
#         # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
#         # List[LSTMState]: One state per layer
#         output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
#         output = input
#         # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
#         i = 0
#         for rnn_layer in self.layers:
#             state = states[i]
#             output, out_state = rnn_layer(output, state)
#             # Apply the dropout layer except the last layer
#             if i < self.num_layers - 1:
#                 output = self.dropout_layer(output)
#             output_states += [out_state]
#             i += 1
#         return output, output_states
#
#
# def flatten_states(states):
#     states = list(zip(*states))
#     assert len(states) == 2
#     return [torch.stack(state) for state in states]
#
#
# def double_flatten_states(states):
#     # XXX: Can probably write this in a nicer way
#     states = flatten_states([flatten_states(inner) for inner in states])
#     return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]
#
#
# def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
#     inp = torch.randn(seq_len, batch, input_size)
#     state = LSTMState(torch.randn(batch, hidden_size),
#                       torch.randn(batch, hidden_size))
#     rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
#     out, out_state = rnn(inp, state)
#
#     # Control: pytorch native LSTM
#     lstm = nn.LSTM(input_size, hidden_size, 1)
#     lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
#     for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
#         assert lstm_param.shape == custom_param.shape
#         with torch.no_grad():
#             lstm_param.copy_(custom_param)
#     lstm_out, lstm_out_state = lstm(inp, lstm_state)
#
#     assert (out - lstm_out).abs().max() < 1e-5
#     assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
#     assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5
#
#
# def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size,
#                             num_layers):
#     inp = torch.randn(seq_len, batch, input_size)
#     states = [LSTMState(torch.randn(batch, hidden_size),
#                         torch.randn(batch, hidden_size))
#               for _ in range(num_layers)]
#     rnn = script_lstm(input_size, hidden_size, num_layers)
#     out, out_state = rnn(inp, states)
#     custom_state = flatten_states(out_state)
#
#     # Control: pytorch native LSTM
#     lstm = nn.LSTM(input_size, hidden_size, num_layers)
#     lstm_state = flatten_states(states)
#     for layer in range(num_layers):
#         custom_params = list(rnn.parameters())[4 * layer: 4 * (layer + 1)]
#         for lstm_param, custom_param in zip(lstm.all_weights[layer],
#                                             custom_params):
#             assert lstm_param.shape == custom_param.shape
#             with torch.no_grad():
#                 lstm_param.copy_(custom_param)
#     lstm_out, lstm_out_state = lstm(inp, lstm_state)
#
#     assert (out - lstm_out).abs().max() < 1e-5
#     assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
#     assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5
#
#
# def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size,
#                                   num_layers):
#     inp = torch.randn(seq_len, batch, input_size)
#     states = [[LSTMState(torch.randn(batch, hidden_size),
#                          torch.randn(batch, hidden_size))
#                for _ in range(2)]
#               for _ in range(num_layers)]
#     rnn = script_lstm(input_size, hidden_size, num_layers, bidirectional=True)
#     out, out_state = rnn(inp, states)
#     custom_state = double_flatten_states(out_state)
#
#     # Control: pytorch native LSTM
#     lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
#     lstm_state = double_flatten_states(states)
#     for layer in range(num_layers):
#         for direct in range(2):
#             index = 2 * layer + direct
#             custom_params = list(rnn.parameters())[4 * index: 4 * index + 4]
#             for lstm_param, custom_param in zip(lstm.all_weights[index],
#                                                 custom_params):
#                 assert lstm_param.shape == custom_param.shape
#                 with torch.no_grad():
#                     lstm_param.copy_(custom_param)
#     lstm_out, lstm_out_state = lstm(inp, lstm_state)
#
#     assert (out - lstm_out).abs().max() < 1e-5
#     assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
#     assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5
#
#
# def test_script_stacked_lstm_dropout(seq_len, batch, input_size, hidden_size,
#                                      num_layers):
#     inp = torch.randn(seq_len, batch, input_size)
#     states = [LSTMState(torch.randn(batch, hidden_size),
#                         torch.randn(batch, hidden_size))
#               for _ in range(num_layers)]
#     rnn = script_lstm(input_size, hidden_size, num_layers, dropout=True)
#
#     # just a smoke test
#     out, out_state = rnn(inp, states)
#
#
# def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size,
#                                num_layers):
#     inp = torch.randn(seq_len, batch, input_size)
#     states = [LSTMState(torch.randn(batch, hidden_size),
#                         torch.randn(batch, hidden_size))
#               for _ in range(num_layers)]
#     rnn = script_lnlstm(input_size, hidden_size, num_layers)
#
#     # just a smoke test
#     out, out_state = rnn(inp, states)
#
#
# test_script_rnn_layer(5, 2, 3, 7)
# test_script_stacked_rnn(5, 2, 3, 7, 4)
# test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
# test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
# test_script_stacked_lnlstm(5, 2, 3, 7, 4)
