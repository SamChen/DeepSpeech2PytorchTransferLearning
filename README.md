# DeepSpeech2 on Pytorch 

* This repo implement a pytorch-based DeepSpeech2 that can load the DeepSpeech2 model which is pretrained on PaddlePaddle-v0.12. 

## Table of Contents
- [DeepSpeech2 on Pytorch](#deepspeech2-on-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Getting Started](#getting-started)
    - [Structure](#structure)
    - [Manifest for data](#manifest-for-data)
    - [Configuration](#configuration)
      - [Data Augmentation](#data-augmentation)
      - [Experiment setting](#experiment-setting)
  - [Example](#example)
    - [Creating Example](#creating-example)
    - [Download DS2 model and Language Model](#download-ds2-model-and-language-model)
    - [Training a model](#training-a-model)
    - [Inference and Evaluation](#inference-and-evaluation)
  - [Code Structure](#code-structure)
  - [Language Model](#language-model)
    - [English LM](#english-lm)
  - [Released Models](#released-models)
    - [Speech Model Adatped](#speech-model-adatped)
    - [Language Model Released](#language-model-released)
  - [Future Work](#future-work)
    - [Compute Mean & Stddev for Normalizer](#compute-mean--stddev-for-normalizer)
    - [Build Vocabulary](#build-vocabulary)
    - [Acceleration with Multi-GPUs](#acceleration-with-multi-gpus)

## Installation

### Prerequisites
tested on Python 3.7 with pytorch 1.4: 
  - Ubuntu19.10 (GPU and CPU)
  - MacOS10.15 (CPU only)

### Setup
1. Make sure these libraries or tools installed: `pkg-config`, `flac`, `ogg`, `vorbis`, `boost` and `swig`, e.g. installing them via `apt-get`:

```bash
sudo apt-get install -y pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python-dev
```
or, installing them via `homebrew` or `linuxbrew`:
```bash
brew install flac libogg libvorbis boost pkg-config eigen
```

2. Run `setup.sh` to install all python packages and compile a C-based decoder. 

## Getting Started
### Structure 
For experiment purpose, the folder structure is data-oriented. `conf` stores all configurations for the training. And, `exps` and `tensorboard` store all experiment result and intermediat data, including corresponding configurations. Experiments are named by the starting time. 

```
├── conf
│   ├── augmentation.config
│   └── experiment.yaml
├── data
│   ├── ...
│   └── manifest in csv format
├── decoded
├── exps
│   ├── 200324-15:55:40
│   ├── ...
│   └── 200411-22:28:15
└── tensorboard
    ├── 200324-15:46:24
    ├── ...
    └── 200411-22:28:15
```


### Manifest for data
Our pytorch version takes a `.csv` file as its data set interface. This csv file should contain following columns: 

| Calumn name |                                                                     Description |
| :---------- | ------------------------------------------------------------------------------: |
| uttid       |                                                                    utterance ID |
| st          | start time of the utterance (Required, if set `segmented=False` for dataloader). |
| et          |   end time of the utterance (Required, if set `segmented=False` for dataloader.) |
| text        |                                                           Manual transcription. |
| audio_path  |                                                   absolute path of audio recordings|
| duration    |                                                          duration of the audio. |

### Configuration
Currently include two configurations. One for data augmentation and the other is for training process.

#### Data Augmentation 
We kept the original code for data augmentation and feature extraction. Six optional augmentation components are provided to be selected, configured and inserted into the processing pipeline.

  - Volume Perturbation
  - Speed Perturbation
  - Shifting Perturbation
  - Online Bayesian normalization
  - Noise Perturbation (need background noise audio files)
  - Impulse Response (need impulse audio files)

In order to inform the trainer of what augmentation components are needed and what their processing orders are, it is required to prepare in advance an *augmentation configuration file* in [JSON](http://www.json.org/) format. For example:

```
[{
    "type": "speed",
    "params": {"min_speed_rate": 0.95,
               "max_speed_rate": 1.05},
    "prob": 0.6
},
{
    "type": "shift",
    "params": {"min_shift_ms": -5,
               "max_shift_ms": 5},
    "prob": 0.8
}]
```

If the dataloader is set by this example configuration, every audio clip which is loader by the dataloader will be processed by two steps: First, with 60% of chance, it will be speed perturbed with a uniformly random sampled speed-rate between 0.95 and 1.05. And second, with 80% of chance it will be shifted in time with a random sampled offset between -5 ms and 5 ms. The newly synthesized audio clip will be feed into the feature extractor for further training.

For other configuration examples, please refer to `notebooks/example/conf/augmentation.conf`.

Be careful when utilizing the data augmentation technique, as improper augmentation will do harm to the training, due to the enlarged train-test gap.

#### Experiment setting
`experiment.yaml` stores all configurations for training a model including the data augmentation which is introduced in the next section. Please check `notebooks/example/conf/experiment.yaml` for more details.

## Example
### Creating Example
Several audio recordings from Voxforge are stored in `notebooks/example/audio`. Since the manifest requirements absolution path for each recording. The user has to run `generate_manifest.py` to generate example manifest. In this example, `train set`, `val set` and `test set` are pointed to the `example.csv`. 
```
cd notebooks/example
python generate_manifest.py
```

### Download DS2 model and Language Model
Download DeepSpeech model and Language Model from the following [link](https://ohsu.app.box.com/folder/110543895998). This should take a while as the language model is huge. In the meantime, change `ds2_model_path`, `language_model_path`, `vocab_filepath` and `mean_std_filepath` accordingly. 

### Training a model
<font color=red> Training on CPU is runnable, but it is super slow.</font> Run following code for training:
```
cd notebooks
python deepspeech_train.py example/conf/experiment.yaml
```


### Inference and Evaluation
Run following code for testing:
```
cd notebooks
python deepspeech_test.py example/conf/experiment.yaml example/decoded
```

## Code Structure
Important codes are distributed to five folders. Data augmentation and decoder comes from PaddlePaddle version. Rewrite the network part.  
```
├── data_utils: code for data augmentation and dataloader
├── decoders: code for decoding
├── model_utils: code for networds 
├── notebooks: example and other test code
└── utils: yaml reader
```


## Language Model

### English LM
The English corpus is from the [Common Crawl Repository](http://commoncrawl.org) and you can download it from [statmt](http://data.statmt.org/ngrams/deduped_en). We use part en.00 to train our English language model. There are some preprocessing steps before training:

  * Characters not in \[A-Za-z0-9\s'\] (\s represents whitespace characters) are removed and Arabic numbers are converted to English numbers like 1000 to one thousand.
  * Repeated whitespace characters are squeezed to one and the beginning whitespace characters are removed. Notice that all transcriptions are lowercase, so all characters are converted to lowercase.
  * Top 400,000 most frequent words are selected to build the vocabulary and the rest are replaced with 'UNKNOWNWORD'.

## Released Models
Currently, only we only transform the BaiduEN8k. The model trained on Librispeech will be our future work.

### Speech Model Adatped 
Currently, we only adapted the BaiduEN8k Model.  

| Language |                                       Model Name                                        |         Training Data          | Hours of Speech |
| :------: | :-------------------------------------------------------------------------------------: | :----------------------------: | --------------: |
| English  | [BaiduEN8k Model](https://github.com/SamChen/DeepSpeech2PytorchTransferLearning/releases/download/v0.0.1/baidu_en8k_model.tar.gz) | Baidu Internal English Dataset |          8628 h |

### Language Model Released

|                                      Language Model                                      |                                                     Training Data                                                      | Token-based |   Size | Descriptions                                                                                         |
| :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: | :---------: | -----: | :--------------------------------------------------------------------------------------------------- |
| [English LM](https://github.com/SamChen/DeepSpeech2PytorchTransferLearning/releases) | [CommonCrawl(en.00)](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | Word-based  | 8.3 GB | Pruned with 0 1 1 1 1; <br/> About 1.85 billion n-grams; <br/> 'trie'  binary with '-a 22 -q 8 -b 8' |

## Future Work
### Compute Mean & Stddev for Normalizer
<font color=red>
Have not modified for the new manifest structure yet.
</font>

To perform z-score normalization (zero-mean, unit stddev) upon audio features, we have to estimate in advance the mean and standard deviation of the features, with some training samples:

```bash
python tools/compute_mean_std.py \
--num_samples 2000 \
--specgram_type linear \
--manifest_paths data/librispeech/manifest.train \
--output_path data/librispeech/mean_std.npz
```

It will compute the mean and standard deviation of power spectrum feature with 2000 random sampled audio clips listed in `data/librispeech/manifest.train` and save the results to `data/librispeech/mean_std.npz` for further usage.


### Build Vocabulary
<font color=red>
Have not modified for the new manifest structure yet.
</font>

A vocabulary of possible characters is required to convert the transcription into a list of token indices for training, and in decoding, to convert from a list of indices back to text again. Such a character-based vocabulary can be built with `tools/build_vocab.py`.

```bash
python tools/build_vocab.py \
--count_threshold 0 \
--vocab_path data/librispeech/eng_vocab.txt \
--manifest_paths data/librispeech/manifest.train
```

It will write a vocabuary file `data/librispeeech/eng_vocab.txt` with all transcription text in `data/librispeech/manifest.train`, without vocabulary truncation (`--count_threshold 0`).

### Acceleration with Multi-GPUs
<font color=red>
Not supported yet. 
</font>



