#!/usr/bin/env python
import numpy as np

def load_parameter(file_name):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32)

def load_weights_hiddenlayers():
    conv0_weights   = load_parameter("../models/baidu_en8k/params/___conv_0__.w0")
    conv0_weights   = conv0_weights.reshape(32, 1, 41, 11)
    conv0_weights   = np.transpose(conv0_weights, (0, 1, 3, 2))
    conv0_bn_mean   = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.w1")
    conv0_bn_var    = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.w2")
    conv0_bn_gamma  = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.w0")
    conv0_bn_beta   = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.wbias")


    conv1_weights   = load_parameter("../models/baidu_en8k/params/___conv_1__.w0")
    conv1_weights   = conv1_weights.reshape(32, 32, 21, 11)
    conv1_weights   = np.transpose(conv1_weights, (0, 1, 3, 2))
    conv1_bn_mean   = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.w1")
    conv1_bn_var    = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.w2")
    conv1_bn_gamma  = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.w0")
    conv1_bn_beta   = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.wbias")

    # gru0
    bigru0_directions_0_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_0__.w0")
    bigru0_directions_0_cell_weight_i         = bigru0_directions_0_cell_weight_i.reshape(41 * 32, 1024*3)
    bigru0_directions_0_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_0__.w0")
    w_u_r = bigru0_directions_0_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru0_directions_0_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru0_directions_0_cell_weight_h = np.concatenate([w_u_r,w_c], 1)
    bigru0_directions_0_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_0__.wbias")
    bigru0_directions_0_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.wbias")
    bigru0_directions_0_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.w0")
    bigru0_directions_0_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.w1")
    bigru0_directions_0_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.w2")

    bigru0_directions_1_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_1__.w0")
    bigru0_directions_1_cell_weight_i         = bigru0_directions_1_cell_weight_i.reshape(41 * 32, 1024*3)
    bigru0_directions_1_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_1__.w0")
    w_u_r = bigru0_directions_1_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c = bigru0_directions_1_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru0_directions_1_cell_weight_h = np.concatenate([w_u_r,w_c], 1)
    bigru0_directions_1_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_1__.wbias")
    bigru0_directions_1_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.wbias")
    bigru0_directions_1_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.w0")
    bigru0_directions_1_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.w1")
    bigru0_directions_1_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.w2")


    # gru1
    bigru1_directions_0_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_2__.w0")
    bigru1_directions_0_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_2__.w0")
    bigru1_directions_0_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_2__.wbias")
    bigru1_directions_0_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.wbias")
    bigru1_directions_0_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.w0")
    bigru1_directions_0_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.w1")
    bigru1_directions_0_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.w2")
    bigru1_directions_0_cell_weight_i         = bigru1_directions_0_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru1_directions_0_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru1_directions_0_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru1_directions_0_cell_weight_h = np.concatenate([w_u_r,w_c], 1)


    bigru1_directions_1_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_3__.w0")
    bigru1_directions_1_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_3__.w0")
    bigru1_directions_1_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_3__.wbias")
    bigru1_directions_1_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.wbias")
    bigru1_directions_1_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.w0")
    bigru1_directions_1_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.w1")
    bigru1_directions_1_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.w2")
    bigru1_directions_1_cell_weight_i         = bigru1_directions_1_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru1_directions_1_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru1_directions_1_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru1_directions_1_cell_weight_h = np.concatenate([w_u_r,w_c], 1)

    # gru2
    bigru2_directions_0_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_4__.w0")
    bigru2_directions_0_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_4__.w0")
    bigru2_directions_0_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_4__.wbias")
    bigru2_directions_0_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.wbias")
    bigru2_directions_0_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.w0")
    bigru2_directions_0_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.w1")
    bigru2_directions_0_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.w2")
    bigru2_directions_0_cell_weight_i         = bigru2_directions_0_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru2_directions_0_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru2_directions_0_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru2_directions_0_cell_weight_h = np.concatenate([w_u_r,w_c], 1)


    bigru2_directions_1_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_5__.w0")
    bigru2_directions_1_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_5__.w0")
    bigru2_directions_1_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_5__.wbias")
    bigru2_directions_1_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.wbias")
    bigru2_directions_1_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.w0")
    bigru2_directions_1_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.w1")
    bigru2_directions_1_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.w2")
    bigru2_directions_1_cell_weight_i         = bigru2_directions_1_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru2_directions_1_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru2_directions_1_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru2_directions_1_cell_weight_h = np.concatenate([w_u_r,w_c], 1)
    pretrained_weights = {
        "conv_bn_mask0.conv.weight"                 : conv0_weights,
        "conv_bn_mask0.bn.weight"                   : conv0_bn_gamma,
        "conv_bn_mask0.bn.bias"                     : conv0_bn_beta,
        "conv_bn_mask0.bn.running_mean"             : conv0_bn_mean,
        "conv_bn_mask0.bn.running_var"              : conv0_bn_var ,
        "conv_bn_mask1.conv.weight"                 : conv1_weights,
        "conv_bn_mask1.bn.weight"                   : conv1_bn_gamma,
        "conv_bn_mask1.bn.bias"                     : conv1_bn_beta,
        "conv_bn_mask1.bn.running_mean"             : conv1_bn_mean,
        "conv_bn_mask1.bn.running_var"              : conv1_bn_var ,
        "bigru0.rnn.directions.0.cell.weight_h"     : bigru0_directions_0_cell_weight_h,
        "bigru0.rnn.directions.0.cell.bias"         : bigru0_directions_0_cell_bias,
        "bigru0.f_weight_i"                         : bigru0_directions_0_cell_weight_i,
        "bigru0.f_bn.bias"                          : bigru0_directions_0_cell_bn_bias        ,
        "bigru0.f_bn.weight"                        : bigru0_directions_0_cell_bn_weight      ,
        "bigru0.f_bn.running_mean"                  : bigru0_directions_0_cell_bn_running_mean,
        "bigru0.f_bn.running_var"                   : bigru0_directions_0_cell_bn_running_var ,
        "bigru0.rnn.directions.1.cell.weight_h"     : bigru0_directions_1_cell_weight_h,
        "bigru0.rnn.directions.1.cell.bias"         : bigru0_directions_1_cell_bias,
        "bigru0.b_weight_i"                         : bigru0_directions_1_cell_weight_i,
        "bigru0.b_bn.bias"                          : bigru0_directions_1_cell_bn_bias        ,
        "bigru0.b_bn.weight"                        : bigru0_directions_1_cell_bn_weight      ,
        "bigru0.b_bn.running_mean"                  : bigru0_directions_1_cell_bn_running_mean,
        "bigru0.b_bn.running_var"                   : bigru0_directions_1_cell_bn_running_var ,

        "bigru1.rnn.directions.0.cell.weight_h"     : bigru1_directions_0_cell_weight_h,
        "bigru1.rnn.directions.0.cell.bias"         : bigru1_directions_0_cell_bias,
        "bigru1.f_weight_i"                         : bigru1_directions_0_cell_weight_i,
        "bigru1.f_bn.bias"                          : bigru1_directions_0_cell_bn_bias        ,
        "bigru1.f_bn.weight"                        : bigru1_directions_0_cell_bn_weight      ,
        "bigru1.f_bn.running_mean"                  : bigru1_directions_0_cell_bn_running_mean,
        "bigru1.f_bn.running_var"                   : bigru1_directions_0_cell_bn_running_var ,
        "bigru1.rnn.directions.1.cell.weight_h"     : bigru1_directions_1_cell_weight_h,
        "bigru1.rnn.directions.1.cell.bias"         : bigru1_directions_1_cell_bias,
        "bigru1.b_weight_i"                         : bigru1_directions_1_cell_weight_i,
        "bigru1.b_bn.bias"                          : bigru1_directions_1_cell_bn_bias        ,
        "bigru1.b_bn.weight"                        : bigru1_directions_1_cell_bn_weight      ,
        "bigru1.b_bn.running_mean"                  : bigru1_directions_1_cell_bn_running_mean,
        "bigru1.b_bn.running_var"                   : bigru1_directions_1_cell_bn_running_var ,

        "bigru2.rnn.directions.0.cell.weight_h"     : bigru2_directions_0_cell_weight_h,
        "bigru2.rnn.directions.0.cell.bias"         : bigru2_directions_0_cell_bias,
        "bigru2.f_weight_i"                         : bigru2_directions_0_cell_weight_i,
        "bigru2.f_bn.bias"                          : bigru2_directions_0_cell_bn_bias        ,
        "bigru2.f_bn.weight"                        : bigru2_directions_0_cell_bn_weight      ,
        "bigru2.f_bn.running_mean"                  : bigru2_directions_0_cell_bn_running_mean,
        "bigru2.f_bn.running_var"                   : bigru2_directions_0_cell_bn_running_var ,
        "bigru2.rnn.directions.1.cell.weight_h"     : bigru2_directions_1_cell_weight_h,
        "bigru2.rnn.directions.1.cell.bias"         : bigru2_directions_1_cell_bias,
        "bigru2.b_weight_i"                         : bigru2_directions_1_cell_weight_i,
        "bigru2.b_bn.bias"                          : bigru2_directions_1_cell_bn_bias        ,
        "bigru2.b_bn.weight"                        : bigru2_directions_1_cell_bn_weight      ,
        "bigru2.b_bn.running_mean"                  : bigru2_directions_1_cell_bn_running_mean,
        "bigru2.b_bn.running_var"                   : bigru2_directions_1_cell_bn_running_var
    }
    return pretrained_weights


def load_weights_bottleneck():
    battleneck_weight = load_parameter("../models/baidu_en8k/params/___fc_layer_6__.w0")
    battleneck_weight = battleneck_weight.reshape(2048, 29).transpose(1,0)
    battleneck_bias   = load_parameter("../models/baidu_en8k/params/___fc_layer_6__.wbias")
    pretrained_weights = {
        "bottleneck.weight"                         : battleneck_weight,
        "bottleneck.bias"                           : battleneck_bias,
    }
    return pretrained_weights


def load_weights():
    conv0_weights   = load_parameter("../models/baidu_en8k/params/___conv_0__.w0")
    conv0_weights   = conv0_weights.reshape(32, 1, 41, 11)
    conv0_weights   = np.transpose(conv0_weights, (0, 1, 3, 2))
    conv0_bn_mean   = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.w1")
    conv0_bn_var    = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.w2")
    conv0_bn_gamma  = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.w0")
    conv0_bn_beta   = load_parameter("../models/baidu_en8k/params/___batch_norm_0__.wbias")


    conv1_weights   = load_parameter("../models/baidu_en8k/params/___conv_1__.w0")
    conv1_weights   = conv1_weights.reshape(32, 32, 21, 11)
    conv1_weights   = np.transpose(conv1_weights, (0, 1, 3, 2))
    conv1_bn_mean   = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.w1")
    conv1_bn_var    = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.w2")
    conv1_bn_gamma  = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.w0")
    conv1_bn_beta   = load_parameter("../models/baidu_en8k/params/___batch_norm_1__.wbias")

    # gru0
    bigru0_directions_0_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_0__.w0")
    bigru0_directions_0_cell_weight_i         = bigru0_directions_0_cell_weight_i.reshape(41 * 32, 1024*3)
    bigru0_directions_0_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_0__.w0")
    w_u_r = bigru0_directions_0_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru0_directions_0_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru0_directions_0_cell_weight_h = np.concatenate([w_u_r,w_c], 1)
    bigru0_directions_0_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_0__.wbias")
    bigru0_directions_0_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.wbias")
    bigru0_directions_0_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.w0")
    bigru0_directions_0_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.w1")
    bigru0_directions_0_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_2__.w2")

    bigru0_directions_1_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_1__.w0")
    bigru0_directions_1_cell_weight_i         = bigru0_directions_1_cell_weight_i.reshape(41 * 32, 1024*3)
    bigru0_directions_1_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_1__.w0")
    w_u_r = bigru0_directions_1_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c = bigru0_directions_1_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru0_directions_1_cell_weight_h = np.concatenate([w_u_r,w_c], 1)
    bigru0_directions_1_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_1__.wbias")
    bigru0_directions_1_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.wbias")
    bigru0_directions_1_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.w0")
    bigru0_directions_1_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.w1")
    bigru0_directions_1_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_3__.w2")


    # gru1
    bigru1_directions_0_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_2__.w0")
    bigru1_directions_0_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_2__.w0")
    bigru1_directions_0_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_2__.wbias")
    bigru1_directions_0_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.wbias")
    bigru1_directions_0_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.w0")
    bigru1_directions_0_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.w1")
    bigru1_directions_0_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_4__.w2")
    bigru1_directions_0_cell_weight_i         = bigru1_directions_0_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru1_directions_0_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru1_directions_0_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru1_directions_0_cell_weight_h = np.concatenate([w_u_r,w_c], 1)


    bigru1_directions_1_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_3__.w0")
    bigru1_directions_1_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_3__.w0")
    bigru1_directions_1_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_3__.wbias")
    bigru1_directions_1_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.wbias")
    bigru1_directions_1_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.w0")
    bigru1_directions_1_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.w1")
    bigru1_directions_1_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_5__.w2")
    bigru1_directions_1_cell_weight_i         = bigru1_directions_1_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru1_directions_1_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru1_directions_1_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru1_directions_1_cell_weight_h = np.concatenate([w_u_r,w_c], 1)

    # gru2
    bigru2_directions_0_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_4__.w0")
    bigru2_directions_0_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_4__.w0")
    bigru2_directions_0_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_4__.wbias")
    bigru2_directions_0_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.wbias")
    bigru2_directions_0_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.w0")
    bigru2_directions_0_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.w1")
    bigru2_directions_0_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_6__.w2")
    bigru2_directions_0_cell_weight_i         = bigru2_directions_0_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru2_directions_0_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru2_directions_0_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru2_directions_0_cell_weight_h = np.concatenate([w_u_r,w_c], 1)


    bigru2_directions_1_cell_weight_i         = load_parameter("../models/baidu_en8k/params/___fc_layer_5__.w0")
    bigru2_directions_1_cell_weight_h         = load_parameter("../models/baidu_en8k/params/___gru_5__.w0")
    bigru2_directions_1_cell_bias             = load_parameter("../models/baidu_en8k/params/___gru_5__.wbias")
    bigru2_directions_1_cell_bn_bias          = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.wbias")
    bigru2_directions_1_cell_bn_weight        = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.w0")
    bigru2_directions_1_cell_bn_running_mean  = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.w1")
    bigru2_directions_1_cell_bn_running_var   = load_parameter("../models/baidu_en8k/params/___batch_norm_7__.w2")
    bigru2_directions_1_cell_weight_i         = bigru2_directions_1_cell_weight_i.reshape(2048, 1024*3)
    w_u_r = bigru2_directions_1_cell_weight_h.flatten()[:1024*1024*2].reshape(1024,1024*2)
    w_c   = bigru2_directions_1_cell_weight_h.flatten()[1024*1024*2:].reshape(1024,1024)
    bigru2_directions_1_cell_weight_h = np.concatenate([w_u_r,w_c], 1)

    battleneck_weight                          = load_parameter("../models/baidu_en8k/params/___fc_layer_6__.w0")
    battleneck_weight                          = battleneck_weight.reshape(2048, 29).transpose(1,0)
    battleneck_bias                            = load_parameter("../models/baidu_en8k/params/___fc_layer_6__.wbias")

    pretrained_weights = {   "conv_bn_mask0.conv.weight"                 : conv0_weights,
                             "conv_bn_mask0.bn.weight"                   : conv0_bn_gamma,
                             "conv_bn_mask0.bn.bias"                     : conv0_bn_beta,
                             "conv_bn_mask0.bn.running_mean"             : conv0_bn_mean,
                             "conv_bn_mask0.bn.running_var"              : conv0_bn_var ,
                             "conv_bn_mask1.conv.weight"                 : conv1_weights,
                             "conv_bn_mask1.bn.weight"                   : conv1_bn_gamma,
                             "conv_bn_mask1.bn.bias"                     : conv1_bn_beta,
                             "conv_bn_mask1.bn.running_mean"             : conv1_bn_mean,
                             "conv_bn_mask1.bn.running_var"              : conv1_bn_var ,


                             "bigru0.rnn.directions.0.cell.weight_h"     : bigru0_directions_0_cell_weight_h,
                             "bigru0.rnn.directions.0.cell.bias"         : bigru0_directions_0_cell_bias,
                             "bigru0.f_weight_i"                         : bigru0_directions_0_cell_weight_i,
                             "bigru0.f_bn.bias"                          : bigru0_directions_0_cell_bn_bias        ,
                             "bigru0.f_bn.weight"                        : bigru0_directions_0_cell_bn_weight      ,
                             "bigru0.f_bn.running_mean"                  : bigru0_directions_0_cell_bn_running_mean,
                             "bigru0.f_bn.running_var"                   : bigru0_directions_0_cell_bn_running_var ,
                             "bigru0.rnn.directions.1.cell.weight_h"     : bigru0_directions_1_cell_weight_h,
                             "bigru0.rnn.directions.1.cell.bias"         : bigru0_directions_1_cell_bias,
                             "bigru0.b_weight_i"                         : bigru0_directions_1_cell_weight_i,
                             "bigru0.b_bn.bias"                          : bigru0_directions_1_cell_bn_bias        ,
                             "bigru0.b_bn.weight"                        : bigru0_directions_1_cell_bn_weight      ,
                             "bigru0.b_bn.running_mean"                  : bigru0_directions_1_cell_bn_running_mean,
                             "bigru0.b_bn.running_var"                   : bigru0_directions_1_cell_bn_running_var ,

                             "bigru1.rnn.directions.0.cell.weight_h"     : bigru1_directions_0_cell_weight_h,
                             "bigru1.rnn.directions.0.cell.bias"         : bigru1_directions_0_cell_bias,
                             "bigru1.f_weight_i"                         : bigru1_directions_0_cell_weight_i,
                             "bigru1.f_bn.bias"                          : bigru1_directions_0_cell_bn_bias        ,
                             "bigru1.f_bn.weight"                        : bigru1_directions_0_cell_bn_weight      ,
                             "bigru1.f_bn.running_mean"                  : bigru1_directions_0_cell_bn_running_mean,
                             "bigru1.f_bn.running_var"                   : bigru1_directions_0_cell_bn_running_var ,
                             "bigru1.rnn.directions.1.cell.weight_h"     : bigru1_directions_1_cell_weight_h,
                             "bigru1.rnn.directions.1.cell.bias"         : bigru1_directions_1_cell_bias,
                             "bigru1.b_weight_i"                         : bigru1_directions_1_cell_weight_i,
                             "bigru1.b_bn.bias"                          : bigru1_directions_1_cell_bn_bias        ,
                             "bigru1.b_bn.weight"                        : bigru1_directions_1_cell_bn_weight      ,
                             "bigru1.b_bn.running_mean"                  : bigru1_directions_1_cell_bn_running_mean,
                             "bigru1.b_bn.running_var"                   : bigru1_directions_1_cell_bn_running_var ,

                             "bigru2.rnn.directions.0.cell.weight_h"     : bigru2_directions_0_cell_weight_h,
                             "bigru2.rnn.directions.0.cell.bias"         : bigru2_directions_0_cell_bias,
                             "bigru2.f_weight_i"                         : bigru2_directions_0_cell_weight_i,
                             "bigru2.f_bn.bias"                          : bigru2_directions_0_cell_bn_bias        ,
                             "bigru2.f_bn.weight"                        : bigru2_directions_0_cell_bn_weight      ,
                             "bigru2.f_bn.running_mean"                  : bigru2_directions_0_cell_bn_running_mean,
                             "bigru2.f_bn.running_var"                   : bigru2_directions_0_cell_bn_running_var ,
                             "bigru2.rnn.directions.1.cell.weight_h"     : bigru2_directions_1_cell_weight_h,
                             "bigru2.rnn.directions.1.cell.bias"         : bigru2_directions_1_cell_bias,
                             "bigru2.b_weight_i"                         : bigru2_directions_1_cell_weight_i,
                             "bigru2.b_bn.bias"                          : bigru2_directions_1_cell_bn_bias        ,
                             "bigru2.b_bn.weight"                        : bigru2_directions_1_cell_bn_weight      ,
                             "bigru2.b_bn.running_mean"                  : bigru2_directions_1_cell_bn_running_mean,
                             "bigru2.b_bn.running_var"                   : bigru2_directions_1_cell_bn_running_var ,

                             "bottleneck.weight"                         : battleneck_weight,
                             "bottleneck.bias"                           : battleneck_bias,
                         }

    return pretrained_weights

pretrained_weights = load_weights()
