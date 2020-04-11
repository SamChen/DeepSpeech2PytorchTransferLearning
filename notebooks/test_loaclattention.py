import streamlit as st

import torch
from torch import nn


class local_attention_layer(nn.MultiheadAttention):
    """
    The only difference between the localattention and attention is local attention need to mask all uninterested keys.
    """
    @staticmethod
    def attention_mask(seq_lengths, mask, query_index, window_size):
        """
        Based on the mask of input where padding positions are marked as 0s.
        """
        assert len(seq_lengths) == mask.size(1)

        atten_mask = torch.zeros_like(mask, dtype=mask.dtype)

        left_boundry = query_index
        right_boundry = query_index+ 2*window_size

        if left_boundry < 0:
            raise IndexError("query_index must be positive")

        for i in range(len(seq_lengths)):
            if right_boundry < seq_lengths[i]:
                atten_mask[left_boundry: right_boundry+1, i, :] = True

        atten_mask = atten_mask & mask
        atten_mask = ~atten_mask[:,:,0].transpose(1,0).contiguous()

        return atten_mask

    @staticmethod
    def attention_mask2(seq_lengths, mask, query_index, window_size):
        """
        Based on the mask of input where padding positions are marked as 0s.
        """
        assert len(seq_lengths) == 1
        assert len(seq_lengths) == mask.size(1)

        atten_mask = torch.zeros_like(mask, dtype=mask.dtype)

        left_boundry = query_index
        right_boundry = query_index+ 2*window_size

        if left_boundry < 0:
            raise IndexError("query_index must be positive")

        for i in range(len(seq_lengths)):
            if right_boundry < seq_lengths[i]:
                atten_mask[left_boundry: right_boundry+1, i, :] = 1

        atten_mask = atten_mask * mask
        atten_mask = ~atten_mask[:,:,0].transpose(1,0).contiguous()
        atten_mask = torch.zeros(atten_mask.size(0),dtype=torch.float).masked_fill(atten_mask,float("-inf"))#
        return atten_mask

if __name__ == "__main__":
    st.title('Local Attention Test')

    st.write(" -[] Add unit test for local attention: edge cases(negtive index and window cross the max_length) and general case")
    st.write(" -[] Gradient test: key_padding_mask should performs the same as attn_mask when batch_size is 1")
    st.write(" -[] Full local_attention running: it should be a for loop of querys")

    batch_size = 4
    index=3
    input = torch.rand(8,batch_size,3).requires_grad_()
    output = torch.rand(8,batch_size,3)
    mask = torch.ones_like(input, dtype=torch.bool)
    seq_lengths = []
    for i in range(batch_size):
        mask[8-i:,i] = 0
        seq_lengths.append(8-i)
    input = input * mask
    output = output * mask
    query = torch.rand(1,batch_size,3)

    criteria = nn.MSELoss()

    st.write("## Test local mask for general case and edge case")
    local_att = local_attention_layer(embed_dim=3, num_heads=1, bias=True)
    local_mask = local_attention_layer.attention_mask(seq_lengths, mask, query_index=index, window_size=1)

    st.write("{}".format(local_mask.shape))

    st.write("## Test Local Attention output, use key padding mask")

    # TODO: there might be some problem on using key_padding_mask. Once it is masked, the gradient is nan. Does this influence the behavior?
    local_output, local_weights = local_att.forward(query, key=input, value=input, key_padding_mask=local_mask, need_weights=True, attn_mask=None)
    st.write(local_output.shape)
    local_output.masked_fill_(torch.isnan(local_output), 0.0)

    loss1 = criteria(target=output[index],input=local_output)
    loss1.backward()
    st.write("``` {}".format(local_weights))
    st.write("loss1: {}".format(loss1.cpu().item()))


    st.write("## Test difference of key_padding_mask and attn_mask")
    input = input[:,3,:].detach().contiguous()
    input.unsqueeze_(1)
    input.requires_grad_()
    mask = mask[:,3,:].detach().contiguous()
    mask.unsqueeze_(1)

    query = torch.rand(1,1,3)

    st.write(input.shape)
    st.write(mask.shape)

    local_att2 = local_attention_layer(embed_dim=3, num_heads=1, bias=True)
    local_mask2 = local_attention_layer.attention_mask2([seq_lengths[3]], mask, query_index=index, window_size=1)
    st.write("```\n{}".format(local_mask2))

    local_output2, local_weights2 = local_att.forward(query, key=input, value=input, key_padding_mask=None, need_weights=True, attn_mask=local_mask2)
    st.write(local_weights2)

    loss2 = criteria(target=output[index],input=local_output2)
    loss2.backward()
    st.write("```{}".format(input))
    st.write("```{}".format(input.grad))

