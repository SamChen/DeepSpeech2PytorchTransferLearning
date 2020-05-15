"""Contains DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import logging
import gzip
import copy
import inspect
import numpy as np
from distutils.dir_util import mkpath
from collections import defaultdict
from itertools import compress
import pandas as pd
from torch.utils.data import DataLoader

# try:
from decoders.swig_wrapper import Scorer
from decoders.swig_wrapper import ctc_greedy_decoder
from decoders.swig_wrapper import ctc_beam_search_decoder_batch
# except:
#     from decoders.scorer_deprecated   import Scorer
#     from decoders.decoders_deprecated import ctc_greedy_decoder
#     from decoders.decoders_deprecated import ctc_beam_search_decoder_batch


import torch
import torch.nn as nn
import torch.optim as optim

# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')


class DeepSpeech2Model(object):
    """DeepSpeech2Model class.

    :param vocab_size: Decoding vocabulary size.
    :type vocab_size: int
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward directional RNNs.Notice that
                              for GRU, weight sharing is not supported.
    :type share_rnn_weights: bool
    """

    def __init__(self, model,
                 ds2_model_path,
                 vocab_list,
                 device):
        self.vocab_size = len(vocab_list)
        self.vocab_list = vocab_list
        self.device = device
        self.model = self._create_model(model,
                                        self.vocab_size,
                                        self.device)
        self._load_paddle_pretrained(ds2_model_path)
        self.model.to(self.device)

        self._inferer = None
        self._loss_inferer = None
        self._ext_scorer = None
        # the model only contain 2 Conv layers
        self._num_conv_layers = 2

        self.logger = logging.getLogger("")
        self.logger.setLevel(level=logging.INFO)



    def train(self,
              train_dataset,
              train_batchsize,
              val_dataset,
              val_batchsize,
              collate_fn,
              lr_key,
              exclue_lr_key,
              learning_rate,
              scheduler_gamma,
              gradient_clipping,
              num_passes,
              output_dir,
              writer,
              num_iterations_print=100,
              feeding_dict=None,
              sortN_epoch=0,
              num_workers=10,
              specific_lr_dict = None
              ):
        """Train the model for one epoch

        :param train_batch_reader: Train data reader.
        :type train_batch_reader: callable

        :param dev_batch_reader: Validation data reader.
        :type dev_batch_reader: callable

        :param feeding_dict: Feeding is a map of field name and tuple index
                             of the data that reader returns.
        :type feeding_dict: dict|list

        :param learning_rate: Learning rate for ADAM optimizer.
        :type learning_rate: float

        :param gradient_clipping: Gradient clipping threshold.
        :type gradient_clipping: float

        :param num_passes: Number of training epochs.
        :type num_passes: int

        :param num_iterations_print: Number of training iterations for printing
                                     a training loss.
        :type rnn_iteratons_print: int

        :param output_dir: Directory for saving the model (every pass).
        :type output_dir: basestring
        """

        self.model.train()
        self.logger.info("DNN structure: \n {}\n".format(self.model))
        self.logger.info("Learning rate: \n {}\n".format(learning_rate))
        self.logger.info("scheduler_gamma: \n {}\n".format(scheduler_gamma))
        # prepare model output directory
        assert os.path.exists(output_dir)


        # adapt the feeding dict and reader according to the network
        # adapted_train_batch_reader = self._adapt_data(train_batch_reader)
        # adapted_dev_batch_reader = self._adapt_data(dev_batch_reader)

        # create loss
        # PaddlePaddle DeepSpeech2 use puts the blank at the end
        self.criterion = CTCLoss(blank=self.vocab_size, reduction="mean", zero_infinity=True)

        # optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        tuned_param = {n:p for n, p in self.model.named_parameters() \
                        if any(nd in n for nd in lr_key)}

        if exclue_lr_key:
            tuned_param = {n:tuned_param[n] for n in tuned_param \
                            if any(nd not in n for nd in exclue_lr_key)}

        #TODO: implement a flexible optimizer that can custom learning rate for each layer.
        if specific_lr_dict:
            assert isinstance(specific_lr_dict, dict)

            special_param = []
            # special_param_name = []
            common_param = [{"params":tuned_param[n]} for n in tuned_param \
                            if any(nd not in n for nd in specific_lr_dict)]
            # common_param_name = [n for n in tuned_param \
            #                 if any(nd not in n for nd in specific_lr)]

            
            for n in tuned_param:
                key_loc = [nd in n for nd in specific_lr_dict]
                if any(key_loc):
                    # sooooo ugly!!!
                    special_param.append({"params":tuned_param[n], "lr": list(compress(specific_lr_dict.values(), key_loc))[0]})
                    # special_param_name.append(n)

            optim_param = common_param + special_param
        else:
            optim_param = [{"params":tuned_param[n]} for n in tuned_param]

        self.logger.info("learnable parameter name: {}\n".format(tuned_param.keys()))
        optimizer = optim.AdamW(optim_param, lr=learning_rate)
        # optimizer = optim.Adam(tuned_param.values(), lr=learning_rate,
        #                         betas=(0.9, 0.98), eps=1e-9)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=scheduler_gamma,
                                                     last_epoch=-1)

        val_dataloader = DataLoader(val_dataset, batch_size=val_batchsize,
                                    shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn)

        self.logger.info("Start training process")
        global_iter = 0
        last_lr = learning_rate
        for epoch in range(num_passes):
            shuffle = True if epoch >= sortN_epoch else False
            train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize,
                                          shuffle=shuffle, num_workers=num_workers,
                                          collate_fn=collate_fn)
            for index, batch in enumerate(train_dataloader):
                if self.model.training != True:
                    self.model.train()

                # self.optimizer.zero_grad()
                self.model.zero_grad()
                loss = self.compute_loss(batch)
                if loss != loss:
                    self.logger.info("loss: {} \ninput {}".format(loss.item(), batch["uttid"]))
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(tuned_param.values(), max_norm=gradient_clipping)
                optimizer.step()

                self.logger.debug("epoch{}, global_iter{} train loss: {}".format(epoch, global_iter, loss.item()))
                if global_iter % num_iterations_print == 0:
                    val_loss = self.evaluate(val_dataloader)

                    self.logger.info("epoch: {}, global_iter: {}, last_lr: {},loss/train: {}, loss/val: {}".format(epoch, global_iter, last_lr, loss.item(), val_loss))
                    writer.add_scalar('Loss/train', loss.item(), global_iter)
                    writer.add_scalar('Loss/val', val_loss, global_iter)

                    torch.save(self.model.state_dict(),
                               os.path.join(output_dir, "models/model_{}.pth".format(global_iter)))

                    val_detail = self.decode(val_dataloader)
                    val_detail.to_pickle(os.path.join(output_dir, "vals/val_detail_{}.pkl".format(global_iter)))


                global_iter += 1

            scheduler.step()
            last_lr = scheduler.get_last_lr()

        val_loss = self.evaluate(val_dataloader)
        self.logger.info("global_iter: {}, loss/train: {}, loss/val: {}".format(global_iter, loss.item(), val_loss))
        writer.add_scalar('Loss/train', loss.item(), global_iter)
        writer.add_scalar('Loss/val', val_loss, global_iter)
        torch.save(self.model.state_dict(),
                   os.path.join(output_dir, "models/model_{}.pth".format("final")))


    def decode(self, dataloader):
        outputs = defaultdict(list)
        for i_batch, sample_batched in enumerate(dataloader):
            batch_results = self.infer_batch_probs(infer_data=sample_batched)
            batch_transcripts_beam = self.decode_batch_beam_search(probs_split=batch_results,
                                                                   beam_alpha=2,
                                                                   beam_beta=0.35,
                                                                   beam_size=500,
                                                                   cutoff_prob=1.0,
                                                                   cutoff_top_n=40,
                                                                   num_processes=5)
            outputs["uttid"].extend(sample_batched["uttid"])
            outputs["probs"].extend(batch_results)
            outputs["asr"].extend(batch_transcripts_beam)
            outputs["text"].extend(sample_batched["trans"])

        outputs = pd.DataFrame.from_dict(outputs)
        return outputs

    def evaluate(self, dataloader)->float:
        total_loss = []
        for index, batch in enumerate(dataloader):
            loss = self.compute_loss(batch)
            total_loss.append(loss.item())

            # if loss is NaN
            if loss != loss:
                for i,_ in enumerate(batch["uttid"]):
                    self.logger.debug("uttid: {}, length_spec: {}, text: {}".format(batch["uttid"][i], batch["length_spec"][i], batch["trans"][i]))
            # if loss is inf
            if loss == float('inf'):
                for i,_ in enumerate(batch["uttid"]):
                    self.logger.debug("uttid: {}, length_spec: {}, text: {}".format(batch["uttid"][i], batch["length_spec"][i], batch["trans"][i]))

        return np.mean(total_loss)



    def compute_loss(self, batch):
        self.model.zero_grad()
        batch = self._adapt_data(batch)
        refs = batch["text"]
        length_refs = batch["length_text"]
        flattened_refs = DeepSpeech2Model.flatten_paded_seq(refs, length_refs)
        hyps, length_hyps, other = self.model(batch)
        hyps = hyps[0]
        flattened_refs = flattened_refs.to(self.device)
        # (log_probs, targets) must on the same device
        # input_lengths, target_lengths
        loss = self.criterion(log_probs=hyps,
                              targets=flattened_refs,
                              input_lengths=length_hyps,
                              target_lengths=length_refs)

        if loss != loss:
            self.logger.debug("uttid: {}".format(batch["uttid"]))
            self.logger.debug("length_hyps: {}, length_refs: {}".format(length_hyps, length_refs))
            self.logger.debug("hyps: {}".format(hyps))
            self.logger.debug("other: {}".format(other))
        return loss


    @staticmethod
    def flatten_paded_seq(text, length):
        assert isinstance(text, torch.IntTensor), "{}".format(text.type())
        assert isinstance(length, torch.IntTensor), "{}".format(length.type())
        flattened_text = torch.cat([text[i][:length[i]] for i in range(text.shape[0])])
        return flattened_text

    def infer_batch_probs(self, infer_data):
        """Infer the prob matrices for a batch of speech utterances.

        :param infer_data: List of utterances to infer, with each utterance
                           consisting of a tuple of audio features and
                           transcription text (empty string).
        :type infer_data: list
        :return: List of 2-D probability matrix, and each consists of prob
                 vectors for one speech utterancce.
        :rtype: List of matrix
        """
        if self.model.training:
            self.model.eval()
        # define inferer
        adapted_infer_data = self._adapt_data(infer_data)

        # run inference
        with torch.no_grad():
            infer_results = self.model(adapted_infer_data)
            results, lengths, _ = infer_results

            results = results[0].data.cpu().numpy()
            probs_split = []
            for i in range(results.shape[0]):
                probs_split.append(results[i][:lengths[i]])

        return probs_split

    def decode_batch_greedy(self, probs_split):
        """Decode by best path for a batch of probs matrix input.

        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :return: List of transcription texts.
        :rtype: List of basestring
        """
        results = []
        for i, probs in enumerate(probs_split):
            output_transcription = ctc_greedy_decoder(
                probs_seq=probs, vocabulary=self.vocab_list)
            results.append(output_transcription)
        return results

    def init_ext_scorer(self, beam_alpha, beam_beta, language_model_path):
        """Initialize the external scorer. This is where we use the language model

        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param language_model_path: Filepath for language model. If it is
                                    empty, the external scorer will be set to
                                    None, and the decoding method will be pure
                                    beam search without scorer.
        :type language_model_path: basestring|None
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        """
        if language_model_path != '':
            self.logger.info("begin to initialize the external scorer "
                             "for decoding")
            self._ext_scorer = Scorer(beam_alpha, beam_beta,
                                      language_model_path, self.vocab_list)
            lm_char_based = self._ext_scorer.is_character_based()
            lm_max_order = self._ext_scorer.get_max_order()
            lm_dict_size = self._ext_scorer.get_dict_size()
            self.logger.info("language model: "
                             "is_character_based = %d," % lm_char_based +
                             " max_order = %d," % lm_max_order +
                             " dict_size = %d" % lm_dict_size)
            self.logger.info("end initializing scorer")
        else:
            self._ext_scorer = None
            self.logger.info("no language model provided, "
                             "decoding by pure beam search without scorer.")

    def decode_batch_beam_search(self, probs_split, beam_alpha, beam_beta,
                                 beam_size, cutoff_prob, cutoff_top_n,
                                 num_processes):
        """Decode by beam search for a batch of probs matrix input.
           Beam Search already take language model into account

        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                        characters with highest probs in vocabulary will be
                        used in beam search, default 40.
        :type cutoff_top_n: int
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of basestring
        """
        if self._ext_scorer != None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)
        # beam search decode
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=self.vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self._ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n)

        results = [result[0][1] for result in beam_search_results]
        return results


    def _adapt_data(self, batch):
        """Adapt data according to network struct.

        For each convolution layer in the conv_group, to remove impacts from
        padding data, we can multiply zero to the padding part of the outputs
        of each batch normalization layer. We add a scale_sub_region layer after
        each batch normalization layer to reset the padding data.
        For rnn layers, to remove impacts from padding data, we can truncate the
        padding part before output data feeded into the first rnn layer. We use
        sub_seq layer to achieve this.

        :param data: Data from data_provider.
        :type data: list|function
        :return: Adapted data.
        :rtype: list|function
        """
        assert "length_spec" in batch.keys()
        adapted_batch = batch
        # no padding part
        audio_lens = batch["length_spec"]

        # Stride size for conv0 is (3, 2)
        # Stride size for conv1 to convN is (1, 2)
        # Same as the network, hard-coded here
        valid_w = (audio_lens - 1) // 3 + 1

        mask_length = []
        # adding conv layer making info
        # deepspeech's CNN layer will not thrink after the first layer.
        mask_length.append(valid_w)
        for i in range(self._num_conv_layers - 1):
            mask_length.append(valid_w)
        adapted_batch["cnn_masks"] = mask_length
        return adapted_batch

    def _load_paddle_pretrained(self, model_path):
        """Load pretrained DeepSpeech parameters."""
        assert self.model
        self.model.load_paddle_pretrained(model_path)


    def _create_model(self, model, vocab_size, device):
        """Create data layers and model network."""
        return  model(device)

    def load_weights(self, path):
        """Load weights"""
        pretrained_dict = torch.load(path)
        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_matched_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        self.logger.info("load weights from: {}".format(path))
        self.logger.info("excluded weights: {}".format(set(pretrained_dict.keys())- set(model_dict)))

