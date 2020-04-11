"""Contains data generator for orgnaizing various audio data preprocessing
pipeline and offering data reader interface of PaddlePaddle requirements.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tarfile
import os
import numpy as np
import pandas as pd

from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence

#TODO: add extra functions based on data.py

#TODO: add dummy dataloader for test purpose

class DynamicLengthGenerator(Dataset):
    @staticmethod
    def padding_batch(batch, padding_to=-1):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach. This specific designed for collate_fn in torch.utils.data.DataLoader

        If ``padding_to`` is -1, the maximun shape in the batch will be used
        as the target shape for padding. Otherwise, `padding_to` will be the
        target shape (only refers to the second axis).

        """
        # get target shape
        spec_lengths = [i["specgrams"].shape[1] for i in batch]
        max_length = max(spec_lengths)
        sorted_index = np.argsort(spec_lengths)

        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("If padding_to is not -1, it should be larger "
                                 "than any instance's shape in the batch")
            max_length = padding_to

        # padding
        new_batch = {"uttid":[], "specgrams":[], "text":[], "length_spec":[], "length_text":[], "trans": []}
        for i in reversed(sorted_index):
            sample = batch[i]
            audio = sample["specgrams"]
            text = sample["text"]
            uttid = sample["uttid"]
            trans = sample["trans"]

            new_batch["uttid"].append(uttid)
            new_batch["specgrams"].append(torch.tensor(audio, dtype=torch.float).transpose(1,0))
            new_batch["length_spec"].append(audio.shape[1])
            new_batch["text"].append(torch.tensor(text))
            new_batch["length_text"].append(len(text))
            new_batch["trans"].append(trans)

        temp_padded_specgrams = pad_sequence(new_batch["specgrams"], batch_first=True)
        # make the specgrams fit the CNN layer
        new_batch["specgrams"] = torch.unsqueeze(temp_padded_specgrams, dim=1).type(torch.float32)
        new_batch["text"] = pad_sequence(new_batch["text"], batch_first=True).type(torch.int32)
        new_batch["length_spec"] = torch.tensor(new_batch["length_spec"], dtype=torch.int32)
        new_batch["length_text"] = torch.tensor(new_batch["length_text"], dtype=torch.int32)
        return new_batch


class DummyGenerator(DynamicLengthGenerator):
    def __init__(self, datasize, feat_size, num_class, min_len, max_len):
        self.datasize = datasize
        self.feat_size = feat_size
        self.max_len = max_len
        self.min_len = min_len
        self.num_class = num_class

    def __len__(self):
        return self.datasize


    def __getitem__(self, idx):
        length = np.random.randint(self.min_len, self.max_len)

        specgram = np.random.rand(self.feat_size, length)
        transcript = np.random.randint(0, self.num_class, length//2)

        uttid = 0
        sample = {"uttid": uttid,
                  "specgrams":specgram,
                  "text": transcript}

        return sample


class SpecgramGenerator(DynamicLengthGenerator):
    """audio specgram generator"""
    def __init__(self,
                 manifest,
                 vocab_filepath,
                 mean_std_filepath,
                 augmentation_config='{}',
                 max_duration=float('inf'),
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 specgram_type='linear',
                 use_dB_normalization=True,
                 random_seed=0,
                 keep_transcription_text=False,
                 segmented=False):

            self._max_duration = max_duration
            self._min_duration = min_duration
            self._segmented    = segmented
            self._keep_transcription_text = keep_transcription_text

            if isinstance(manifest, str) and os.path.isfile(manifest):
                self.manifest = pd.read_csv(manifest)
            elif isinstance(manifest, pd.DataFrame):
                self.manifest = manifest
            else:
                raise BaseException("{} is neither an valide path or a pandas DataFrame object".format(manifest))

            # duration filtering
            self.manifest = self.manifest[(self.manifest.duration >= self._min_duration)
                                          & (self.manifest.duration <= self._max_duration)]

            self.manifest = self.manifest.sort_values(by=["duration"],
                                                      ascending=True)

            self._normalizer = FeatureNormalizer(mean_std_filepath)

            self._augmentation_pipeline = AugmentationPipeline(
                augmentation_config=augmentation_config, random_seed=random_seed)

            self._speech_featurizer = SpeechFeaturizer(
                vocab_filepath=vocab_filepath,
                specgram_type=specgram_type,
                stride_ms=stride_ms,
                window_ms=window_ms,
                max_freq=max_freq,
                use_dB_normalization=use_dB_normalization)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        instance = self.manifest.iloc[idx]
        if self._segmented is True:
            specgram, transcript = self.process_utterance(instance["audio_path"],
                                                          instance["text"],
                                                          segments_info=None)
        else:
            specgram, transcript = self.process_utterance(instance["audio_path"],
                                                          instance["text"],
                                                          segments_info={"start":instance["st"], "end":instance["et"]})

        uttid = instance["uttid"]
        sample = {"uttid": uttid,
                  "specgrams":specgram,
                  "text": transcript,
                  "trans": instance["text"]}

        return sample


    def process_utterance(self, audio_file, transcript, uttid=None, segments_info=None):
        """Load, augment, featurize and normalize for speech data.

        :param audio_file: Filepath or file object of audio file.
        :type audio_file: basestring | file
        :param transcript: Transcription text.
        :type transcript: basestring
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
        :rtype: tuple of (2darray, list)
        """

        if isinstance(audio_file, str) and audio_file.startswith('tar:'):
            speech_segment = SpeechSegment.from_file(
                self._subfile_from_tar(audio_file), transcript)
        elif segments_info is None:
            speech_segment = SpeechSegment.from_file(audio_file, transcript)
        else:
            speech_segment = SpeechSegment.slice_from_file(audio_file,
                                                               transcript,
                                                               **segments_info)


        # augment speech. i.e. add noise, speedup
        self._augmentation_pipeline.transform_audio(speech_segment)

        specgram, transcript_part = self._speech_featurizer.featurize(
            speech_segment, self._keep_transcription_text)

        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part


