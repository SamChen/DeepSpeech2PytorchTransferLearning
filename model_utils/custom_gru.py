import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

import numbers
import warnings
from collections import namedtuple, defaultdict
from typing import List, Tuple

def reverse(lst:List[Tensor])-> List[Tensor]:
    return lst[::-1]


class GRUCell(nn.Module):
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
        self.weight_i = Parameter(torch.randn(input_size,  3 * hidden_size))
        self.weight_h = Parameter(torch.randn(hidden_size, 3 * hidden_size))

        # order: b_u, b_r, b_c
        self.bias = Parameter(torch.randn(3 * hidden_size))
        self.bn = nn.BatchNorm1d(3 * hidden_size)

    def forward(self, input, state):
        # type: (Tensor, Tensor) -> (Tensor, List[Tensor])
        # check batch size matches
        for i in range(len(state)):
            assert input.shape[0] == state[i].shape[0], "input batch:{}, {}th hidden element batch:{}".format(input.shape[0], i, state[i].shape[0])
        assert input.shape[1] == self.weight_i.shape[0]
        hx = state[0]

        gates_input = torch.mm(input, self.weight_i)
        gates_input = self.bn(gates_input) # deepspeech only normalize the input part
        gates_input += self.bias
        u,r,c = gates_input.chunk(3, 1)
        u_h,r_h,c_h = self.weight_h.chunk(3,1)


        u += torch.mm(hx, u_h)
        r += torch.mm(hx, r_h)
        u = self.gate_activation(u)
        r = self.gate_activation(r)

        c += torch.mm((r * hx), c_h)
        c = self.state_activation(c)

        # this is how paddlepaddle implement gru.
        # it is different to the general implementation "hy = (u * hx) + ((1.0 -u) * c)"
        hy = ( (1 - u) * hx) + (u * c)

        # keep it as the same format as lstm's ouput for future upgrading
        return hy, [hy,]


class GRU_hiddenCell(nn.Module):
    '''
    This GRU layer leave the input * hidden_i outside.
    This mimics PaddlePaddle's dynamicGRU. Only difference is my implementation based on GRUCell instead of a complete GRU layer
    '''

    def __init__(self, hidden_size, gate_act="sigmoid", state_act="tanh"):
        super(GRU_hiddenCell, self).__init__()
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
        self.weight_h = Parameter(torch.randn(hidden_size, 3 * hidden_size))

        # order: b_u, b_r, b_c
        self.bias = Parameter(torch.randn(3 * hidden_size))

    def forward(self, input: Tensor, state: List[Tensor])-> (Tensor, List[Tensor]):
        # check batch size matches
        assert input.shape[1] == self.weight_h.shape[1], \
            "input's shape ({}) should be the same as the hidden shape ({})".format(input.shape[1], self.weight_h.shape[1])
        hx = state[0]

        gates_input = input + self.bias
        u,r,c = gates_input.chunk(3, 1)
        u_h,r_h,c_h = self.weight_h.chunk(3,1)

        u += torch.mm(hx, u_h)
        r += torch.mm(hx, r_h)
        u = self.gate_activation(u)
        r = self.gate_activation(r)

        c += torch.mm((r * hx), c_h)
        c = self.state_activation(c)

        # this is how paddlepaddle implement gru.
        # it is different to the general implementation "hy = (u * hx) + ((1.0 -u) * c)"
        hy = ( (1 - u) * hx) + (u * c)

        # keep it as the same format as lstm's ouput for future upgrading
        return hy, [hy,]

class RNNLayer(nn.Module):
    def __init__(self, cell, **cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(**cell_args)

    def forward(self, input: PackedSequence, hidden: List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]:
        assert isinstance(input, PackedSequence)

        input, batch_sizes, sorted_indices, unsorted_indices = input
        outputs = []
        input_offset = torch.tensor(0)
        last_batch_size = batch_sizes[0]
        finished_hiddens = [[] for i in range(len(hidden))]

        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                for i in range(len(finished_hiddens)):
                    finished_hiddens[i].append(torch.narrow(hidden[i], 0, -dec, dec))
                    # finished_hiddens[i].append(hidden[i][-dec:])

                for i in range(len(finished_hiddens)):
                    # following way's gradient is closer to pytorch's Offical LSTM than hidden[i] = hidden[i][:-dec]
                    hidden[i] = torch.narrow(hidden[i], 0, 0, batch_size)

            last_batch_size = batch_size

            output, hidden = self.cell(step_input, hidden)
            outputs.append(output)

        # append the finished hidden for the longest sequences in that batch
        # finished_hiddens.append(hidden)
        for i in range(len(finished_hiddens)):
            finished_hiddens[i].append(hidden[i])

        # let the hiddens' order matches the input sequence order,
        # (shortest seq will finish first, while longest seq is the first element is a batch)
        # finished_hiddens.reverse()
        for i in range(len(finished_hiddens)):
            finished_hiddens[i].reverse()

        # hidden = torch.cat(finished_hiddens, 0)
        for i in range(len(finished_hiddens)):
            hidden[i] = torch.cat(finished_hiddens[i], 0)

        outputs = torch.cat(outputs, 0)
        outputs = PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)

        return outputs, hidden

class ReverseRNNLayer(nn.Module):
    def __init__(self, cell, **cell_args):
        super(ReverseRNNLayer, self).__init__()
        self.cell = cell(**cell_args)

    def forward(self, input: PackedSequence, hidden: List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]:
        assert isinstance(input, PackedSequence)

        input, batch_sizes, sorted_indices, unsorted_indices = input
        outputs = []
        input_offset = torch.tensor(input.shape[0])
        last_batch_size = batch_sizes[-1]
        initial_hiddens = hidden

        # hidden = [ [] for i in range(len(hidden)) ]
        hidden = jit.annotate(Tensor, [])
        for i in range(len(initial_hiddens)):
            hidden += [initial_hiddens[i][:last_batch_size]]

        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                for i in range(len(initial_hiddens)):
                    # add more initial hiddens for new sequences's last step
                    hidden[i] = torch.cat((hidden[i], initial_hiddens[i][last_batch_size:batch_size]), 0)

            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            output, hidden = self.cell(step_input, hidden)
            outputs.append(output)

        outputs.reverse()
        outputs = torch.cat(outputs, 0)
        outputs = PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)

        return outputs, hidden


class BidirRNNLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, cell, **cell_args):
        super(BidirRNNLayer, self).__init__()
        self.directions = nn.ModuleList([
            RNNLayer(cell, **cell_args),
            ReverseRNNLayer(cell, **cell_args),
        ])

    def forward_bi(self, input, states: List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]:
        outputs = []
        output_states = defaultdict(list)

        _, batch_size, sorted_indices, unsorted_indices = input[0]
        for index, direction in enumerate(self.directions):
            state = [states[ele][index] for ele in range(len(states))]
            out, out_state = direction(input[index], state)
            outputs.append(out[0])
            for index, value in enumerate(out_state):
                output_states[index].append(torch.unsqueeze(value, 0))

        outputs = nn.utils.rnn.PackedSequence(torch.cat(outputs, -1), batch_size, sorted_indices, unsorted_indices)
        output_states = [torch.cat(output_states[index], 0) for index in output_states]
        return outputs, output_states

    def forward_uni(self, input, states: List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]:
        outputs = []
        output_states = defaultdict(list)

        _, batch_size, sorted_indices, unsorted_indices = input
        for index, direction in enumerate(self.directions):
            state = [states[ele][index] for ele in range(len(states))]
            out, out_state = direction(input, state)
            outputs.append(out[0])
            for index, value in enumerate(out_state):
                output_states[index].append(torch.unsqueeze(value, 0))

        outputs = nn.utils.rnn.PackedSequence(torch.cat(outputs, -1), batch_size, sorted_indices, unsorted_indices)
        output_states = [torch.cat(output_states[index], 0) for index in output_states]
        return outputs, output_states

    def forward(self, input, states: List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]:
        input_type = type(input)
        if input_type is list:
            return self.forward_bi(input, states)

        else:
            return self.forward_uni(input, states)

class DynamicGRUlayer(nn.Module):
    def __init__(self, cell, input_size, hidden_size, gate_act, state_act):
        super(DynamicGRUlayer, self).__init__()

        self.f_weight_i = Parameter(torch.randn(input_size,  3 * hidden_size))
        self.b_weight_i = Parameter(torch.randn(input_size,  3 * hidden_size))
        self.f_bn = nn.BatchNorm1d(3 * hidden_size)
        self.b_bn = nn.BatchNorm1d(3 * hidden_size)
        self.rnn = BidirRNNLayer(cell, hidden_size=hidden_size, gate_act=gate_act, state_act=state_act)

    def forward(self, input: PackedSequence, hidden: List[Tensor]) -> Tuple[PackedSequence, List[Tensor]]:
        assert isinstance(input, PackedSequence)
        x, batch_sizes, sorted_indices, unsorted_indices = input

        f_gates_input = torch.mm(x, self.f_weight_i)
        f_gates_input = self.f_bn(f_gates_input) # deepspeech sequence-wise normalization of input part

        b_gates_input = torch.mm(x, self.b_weight_i)
        b_gates_input = self.b_bn(b_gates_input)

        f_input = PackedSequence(f_gates_input, batch_sizes, sorted_indices, unsorted_indices)
        b_input = PackedSequence(b_gates_input, batch_sizes, sorted_indices, unsorted_indices)

        return self.rnn([f_input, b_input], hidden)


def init_stacked_rnn(num_layers, layer, first_layer_args, other_layer_args):
    # stack multiple RNN layers together
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


# class StackedRNN(jit.ScriptModule):
#     __constants__ = ['layers']  # Necessary for iterating through self.layers
# 
#     def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
#         super(StackedRNN, self).__init__()
#         self.layers = init_stacked_rnn(num_layers, layer, first_layer_args,
#                                         other_layer_args)
# 
#     @jit.script_method
#     def forward(self, input, states):
#         # type: (Tensor, List[Tensor]) -> Tuple[Tensor, List[Tensor]]
#         # List[RNNState]: One state per layer
#         output_states = jit.annotate(List[Tensor], [])
#         output = input
#         # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
#         i = 0
#         for rnn_layer in self.layers:
#             state = states[i]
#             output, out_state = rnn_layer(output, state)
#             output_states += [out_state]
#             i += 1
#         return output, output_states
# 
# 
# # Differs from StackedLSTM in that its forward method takes
# # List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# # except we don't support overriding script methods.
# # https://github.com/pytorch/pytorch/issues/10733
# class StackedRNN2(jit.ScriptModule):
#     __constants__ = ['layers']  # Necessary for iterating through self.layers
# 
#     def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
#         super(StackedRNN2, self).__init__()
#         self.layers = init_stacked_rnn(num_layers, layer, first_layer_args,
#                                         other_layer_args)
# 
#     @jit.script_method
#     def forward(self, input, states):
#         # type: (Tensor, List[List[Tensor]]) -> Tuple[Tensor, List[List[Tensor]]]
#         # List[List[LSTMState]]: The outer list is for layers,
#         #                        inner list is for directions.
#         output_states = jit.annotate(List[List[Tensor]], [])
#         output = input
#         # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
#         i = 0
#         for rnn_layer in self.layers:
#             state = states[i]
#             output, out_state = rnn_layer(output, state)
#             output_states += [out_state]
#             i += 1
#         return output, output_states


