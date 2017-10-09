# AWD-LSTM / AWD-QRNN Language Model

### Averaged Stochastic Gradient Descent with Weight Dropped LSTM or QRNN

This repository contains the code used for [Salesforce Research](https://einstein.ai/)'s [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) paper, originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).
The model comes with instructions to train a word level language model over the Penn Treebank (PTB) and [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2) datasets, though the model is likely extensible to many other datasets.
The model can be composed of an LSTM or a [Quasi-Recurrent Neural Network](https://github.com/salesforce/pytorch-qrnn/) (QRNN) which is two or more times faster than the cuDNN LSTM in this setup while achieving equivalent or better accuracy.

+ Install PyTorch 0.2
+ Run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets
+ Train the base model using `main.py`
+ Finetune the model using `finetune.py`
+ Apply the [continuous cache pointer](https://arxiv.org/abs/1612.04426) to the finetuned model using `pointer.py`

If you use this code or our results in your research, please cite:

```
@article{merityRegOpt,
  title={{Regularizing and Optimizing LSTM Language Models}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1708.02182},
  year={2017}
}
```

## Software Requirements

Python 3 and PyTorch 0.2 are required for the current codebase.

Included below are hyper parameters to get equivalent or better results to those included in the original paper.

If you need to use an earlier version of the codebase, the original code and hyper parameters accessible at the [PyTorch==0.1.12](https://github.com/salesforce/awd-lstm-lm/tree/PyTorch%3D%3D0.1.12) release, with Python 3 and PyTorch 0.1.12 are required.
If you are using Anaconda, installation of PyTorch 0.1.12 can be achieved via:
`conda install pytorch=0.1.12 -c soumith`.

## Experiments

The codebase was modified during the writing of the paper, preventing exact reproduction due to minor differences in random seeds or similar.
We have also seen exact reproduction numbers change when changing underlying GPU.
The guide below produces results largely similar to the numbers reported.

For data setup, run `./getdata.sh`.
This script collects the Mikolov pre-processed Penn Treebank and the WikiText-2 datasets and places them in the `data` directory.

Next, decide whether to use the QRNN or the LSTM as the underlying recurrent neural network model.
The QRNN is many times faster than even Nvidia's cuDNN optimized LSTM (and dozens of times faster than a naive LSTM implementation) yet achieves similar or better results than the LSTM.
At the time of writing, the QRNN models use the same number of parameters and are slightly deeper networks but are two to four times faster per epoch and require less epochs to converge.

The QRNN model uses a QRNN with convolutional size 2 for the first layer, allowing the model to view discrete natural language inputs (i.e. "New York"), while all other layers use a convolutional size of 1.

**Finetuning Note:** Fine-tuning modifies the original saved model `model.pt` file - if you wish to keep the original weights you must copy the file.

**Pointer note:** BPTT just changes the length of the sequence pushed onto the GPU but won't impact the final result.

### Penn Treebank (PTB) with LSTM

The instruction below trains a PTB model that without finetuning achieves perplexities of approximately `61.2` / `58.8` (validation / testing), with finetuning achieves perplexities of approximately `58.8` / `56.5`, and with the continuous cache pointer augmentation achieves perplexities of approximately `53.2` / `52.5`.

+ `python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
+ `python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`
+ `python pointer.py --data data/penn --save PTB.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000`

### Penn Treebank (PTB) with QRNN

The instruction below trains a QRNN model that without finetuning achieves perplexities of approximately `60.6` / `58.3` (validation / testing), with finetuning achieves perplexities of approximately `59.1` / `56.7`, and with the continuous cache pointer augmentation achieves perplexities of approximately `53.4` / `52.6`.

+ `python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB.pt`
+ `python -u finetune.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 404 --dropouti 0.4 --epochs 300 --save PTB.pt`
+ `python pointer.py --model QRNN --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000 --save PTB.pt`

### WikiText-2 (WT2) with LSTM
The instruction below trains a PTB model that without finetuning achieves perplexities of approximately `68.7` / `65.6` (validation / testing), with finetuning achieves perplexities of approximately `67.4` / `64.7`, and with the continuous cache pointer augmentation achieves perplexities of approximately `52.2` / `50.6`.

+ `python main.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ `python finetune.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`
+ `python pointer.py --save WT2.pt --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt 2000 --data data/wikitext-2`

### WikiText-2 (WT2) with QRNN

The instruction below will a QRNN model that without finetuning achieves perplexities of approximately `69.3` / `66.8` (validation / testing), with finetuning achieves perplexities of approximately `68.5` / `65.9`, and with the continuous cache pointer augmentation achieves perplexities of approximately `53.6` / `52.1`.
Better numbers are likely achievable but the hyper parameters have not been extensively searched. These hyper parameters should serve as a good starting point however.

+ `python -u main.py --epochs 500 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550  --nlayers 4 --seed 4002 --model QRNN --wdrop 0.1 --batch_size 40 --save WT2.pt`
+ `python finetune.py --epochs 500 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 4002 --model QRNN --wdrop 0.1 --batch_size 40 --save WT2.pt`
+ `python -u pointer.py --save WT2.pt --model QRNN --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt 2000 --data data/wikitext-2`

## Speed

The default speeds for the models during training on an NVIDIA Quadro GP100:

+ Penn Treebank (batch size 20): LSTM takes 65 seconds per epoch, QRNN takes 28 seconds per epoch
+ WikiText-2 (batch size 20): LSTM takes 180 seconds per epoch, QRNN takes 90 seconds per epoch

The default QRNN models can be far faster than the cuDNN LSTM model, with the speed-ups depending on how much of a bottleneck the RNN is. The majority of the model time above is now spent in softmax or optimization overhead (see [PyTorch QRNN discussion on speed](https://github.com/salesforce/pytorch-qrnn#speed)).

Speeds are approximately three times slower on a K80. On a K80 or other memory cards with less memory you may wish to enable [the cap on the maximum sampled sequence length](https://github.com/salesforce/awd-lstm-lm/blob/ef9369d277f8326b16a9f822adae8480b6d492d0/main.py#L131) to prevent out-of-memory (OOM) errors, especially for WikiText-2.

If speed is a major issue, SGD converges more quickly than our non-monotonically triggered variant of ASGD though achieves a worse overall perplexity.

### Details of the QRNN optimization

For full details, refer to the [PyTorch QRNN repository](https://github.com/salesforce/pytorch-qrnn).

### Details of the LSTM optimization

All the augmentations to the LSTM, including our variant of [DropConnect (Wan et al. 2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) termed weight dropping which adds recurrent dropout, allow for the use of NVIDIA's cuDNN LSTM implementation.
PyTorch will automatically use the cuDNN backend if run on CUDA with cuDNN installed.
This ensures the model is fast to train even when convergence may take many hundreds of epochs.
