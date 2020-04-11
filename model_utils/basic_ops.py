import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, PackedSequence

def iter_packedsequence(ps: PackedSequence):
    data, batchs, _, _ = ps
    st_index = 0
    for indexf, b in enumerate(ps):
        # return Tx1, num_Data, size_F
        yield data[st_index: st_index+b, :, :], indexf
        st_index += b


def packedsequences_merge(packed_data_list):
    packed_data_list = [i.unsqueeze(dim=-2) for i in packed_data_list]
    temp = packed_data_list[0]
    for index, ps in enumerate(packed_data_list):
        assert ps.shape == temp.shape, "{}th shape does not match with its neighbor({}, {})".format(index, ps.shape, temp.shape)
        temp = i

    packed_data_list= torch.cat(packed_data_list, dim=-2)
    packed_data = PackedSequence(packed_data_list, batchs) # TxB,num_Data,size_F

    return packed_data, data
