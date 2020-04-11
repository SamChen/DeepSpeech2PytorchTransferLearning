import unittest
import torch
import torch.nn as nn
import sys
sys.path.append("../model_utils/")
import custom_gru
import copy

SEQ_LENGTH=10000

def set_lstm_weights(model, seed = 9):
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.3)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)
    return model


def exp_initializaton(models):
    return [set_lstm_weights(model) for model in models]


def get_gradient(model, criteria, hyp, ref):
    loss = criteria(hyp, ref)
    loss.backward()


def lstm_comparing(lstm_pt, lstm_custom,
                   initial_hidden_pt, initial_hidden_custom, seq_length=10000,
                   test_name=''):

    for initial_pair in zip(initial_hidden_pt, initial_hidden_custom):
        assert initial_pair[0].requires_grad == True
        assert initial_pair[1].requires_grad == True

    input = torch.rand(3, seq_length, 3)
    ref = torch.empty([3, seq_length], dtype=torch.long).random_(5)

    lstm_pt, lstm_custom = exp_initializaton([lstm_pt, lstm_custom])
    for param_pair in zip(lstm_custom.parameters(), lstm_pt.parameters()):
        assert torch.all(torch.eq(*param_pair))

    # get output
    output_custom = lstm_custom(
        nn.utils.rnn.pack_padded_sequence(input, [seq_length, seq_length, seq_length], batch_first=True),
        initial_hidden_custom
    )
    output_pt = lstm_pt(input, initial_hidden_pt)
    ## assert S_n are the same
    custom = output_custom[1][1].unsqueeze(0)
    pt = output_pt[1][1]
    assert torch.all(torch.eq(custom, pt)), \
           "test name:{} \nmean:{} \nstd: {} \n".format(test_name,
                                                        (custom - pt).mean(),
                                                        (custom - pt).std())
    # print(output_custom[1][1].type(), output_custom[1][1].size())
    # print(output_pt[1][1].type()    , output_pt[1][1].size())
    # assert torch.equal(output_custom[1][1], output_pt[1][1]), \
    #     "test name:{} \nmean:{} \nstd: {} \n".format(test_name,
    #                                                  (output_custom[1][1] - output_pt[1][1]).mean(),
    #                                                  (output_custom[1][1] - output_pt[1][1]).std())

    ## assert h_i are the same
    output_custom = nn.utils.rnn.pad_packed_sequence(output_custom[0], batch_first=True)[0]
    assert torch.all(torch.eq(output_custom, output_pt[0]))


    criteria = nn.CrossEntropyLoss()
    # formatting for feed to the criteria, this is criteria specific
    hyp_custom = torch.transpose(output_custom, 2, 1)
    hyp_pt = output_pt[0]
    hyp_pt = torch.transpose(hyp_pt, 2, 1)

    get_gradient(lstm_custom, criteria, hyp_custom, ref=ref)
    get_gradient(lstm_pt, criteria, hyp_pt, ref=ref)

    # Assert similarity NOT exact the same
    for param_pair in zip(lstm_custom.parameters(), lstm_pt.parameters()):
        assert torch.allclose(param_pair[0].grad, param_pair[1].grad, atol=1e-08), \
            "{}".format(torch.std(param_pair[0].grad - param_pair[1].grad))

    # (similar) assert grad for s_0 and h_0
    for initial_pair in zip(initial_hidden_pt, initial_hidden_custom):
        assert torch.allclose(initial_pair[0].grad, initial_pair[1].grad, atol=1e-08), \
            "{}".format(torch.std(initial_pair[0].grad - initial_pair[1].grad))
    # (same) assert grad for s_0 and h_0
    for pair in zip(initial_hidden_pt, initial_hidden_custom):
        assert torch.all(torch.eq(pair[0].grad, pair[1].grad))


class LSTMCell_format(nn.Module):
    '''
    format the LSTMCell's output as (h, [h,c])
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTMCell_format, self).__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        h, c = self.lstm(input, hidden)
        return h, [h,c]

class RNNforward(unittest.TestCase):
    def test_lstms_single_step(self):
        input = torch.rand(3, 1, 3)
        ref = torch.empty(3, dtype=torch.long).random_(5)
        initial_hidden_pt = [torch.zeros([1, 3, 5]).requires_grad_(), torch.zeros([1, 3, 5]).requires_grad_()]
        initial_hidden_cell_pt = [torch.zeros([3, 5]).requires_grad_() for _ in initial_hidden_pt]

        lstm_pt = nn.LSTM(input_size=3, hidden_size=5, batch_first=True)
        lstm_cell_pt = LSTMCell_format(input_size=3, hidden_size=5)
        lstm_pt, lstm_cell_pt = exp_initializaton([lstm_pt, lstm_cell_pt])

        for param_pair in zip(lstm_pt.parameters(), lstm_cell_pt.parameters()):
            self.assertTrue(torch.all(torch.eq(*param_pair)))


        output_pt = lstm_pt(input, initial_hidden_pt)
        output_cell_pt = lstm_cell_pt(torch.squeeze(input, 1) , initial_hidden_cell_pt)
        for finial_state_pair in zip(output_pt[1], output_cell_pt[1]):
            self.assertTrue(torch.all(torch.eq(finial_state_pair[0], finial_state_pair[1])))


        hyp_pt = output_pt[0]
        hyp_pt = torch.squeeze(hyp_pt, 1)
        hyp_cell_pt = output_cell_pt[0]
        criteria = nn.CrossEntropyLoss()

        get_gradient(lstm_cell_pt, criteria, hyp_cell_pt, ref=ref)
        get_gradient(lstm_pt, criteria, hyp_pt, ref=ref)

        # assert grad for parameter
        for param_pair in zip(lstm_pt.parameters(), lstm_cell_pt.parameters()):
            self.assertTrue(torch.all(torch.eq(param_pair[0].grad, param_pair[1].grad)))

        # assert grad for s_0 and h_0
        for pair in zip(initial_hidden_pt, initial_hidden_cell_pt):
            self.assertTrue(torch.all(torch.eq(pair[0].grad, pair[1].grad)))


    def test_customRNNforward(self):
        test_name = "test_customRNNforward"
        initial_hidden_pt = [torch.zeros([1, 3, 5]).requires_grad_(), torch.zeros([1, 3, 5]).requires_grad_()]
        initial_hidden_custom = [torch.zeros([3, 5]).requires_grad_() for _ in initial_hidden_pt]
        # initialize
        lstm_custom = custom_gru.RNNLayer(LSTMCell_format, input_size=3, hidden_size=5)
        lstm_pt = nn.LSTM(input_size=3, hidden_size=5, batch_first=True)
        lstm_comparing(lstm_pt, lstm_custom,
                       initial_hidden_pt=initial_hidden_pt,
                       initial_hidden_custom=initial_hidden_custom,
                       seq_length=SEQ_LENGTH,
                       test_name=test_name)

    def test_customRNNbidirection(self):
        initial_hidden_pt = [torch.zeros([2, 3, 5]).requires_grad_(), torch.zeros([2, 3, 5]).requires_grad_()]
        initial_hidden_custom = [torch.zeros([2, 3, 5]).requires_grad_() for _ in initial_hidden_pt]

        lstm_custom = custom_gru.BidirRNNLayer(LSTMCell_format, input_size=3, hidden_size=5)
        lstm_pt = nn.LSTM(input_size=3, hidden_size=5, batch_first=True, bidirectional=True)
        lstm_comparing(lstm_pt, lstm_custom,
                       initial_hidden_pt=initial_hidden_pt,
                       initial_hidden_custom=initial_hidden_custom,
                       seq_length=SEQ_LENGTH)


if __name__ == "__main__":
    unittest.main()
