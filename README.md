# AWD-LSTM Language Model

### Averaged Stochastic Gradient Descent with Weight Dropped LSTM

This repository contains the code used for [Salesforce Research](https://einstein.ai/)'s [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) paper, originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).
The model comes with instructions to train a word level language model over the Penn Treebank (PTB) and [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2) datasets, though the model is likely extensible to many other datasets.

+ Install PyTorch 0.1.12_2
+ Run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets
+ Train the base model using `main.py`
+ Finetune the model using `finetune.py`
+ Apply the [continuous cache pointer](http://xxx.lanl.gov/abs/1612.04426) to the finetuned model using `pointer.py`

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

This codebase requires Python 3 and PyTorch 0.1.12_2.
If you are using Anaconda, this can be achieved via:
`conda install pytorch=0.1.12 -c soumith`.

Note the older version of PyTorch - upgrading to later versions would require minor updates and would prevent the exact reproductions of the results below.
Pull requests which update to later PyTorch versions are welcome, especially if they have baseline numbers to report too :)

## Experiments

The codebase was modified during the writing of the paper, preventing exact reproduction due to minor differences in random seeds or similar.
The guide below produces results largely similar to the numbers reported.

For data setup, run `./getdata.sh`.
This script collects the Mikolov pre-processed Penn Treebank and the WikiText-2 datasets and places them in the `data` directory.

**Important:** If you're going to continue experimentation beyond reproduction, comment out the test code and use the validation metrics until reporting your final results.
This is proper experimental practice and is especially important when tuning hyperparameters, such as those used by the pointer.

#### Penn Treebank (PTB)

The instruction below trains a PTB model that without finetuning achieves perplexities of `61.2` / `58.9` (validation / testing), with finetuning achieves perplexities of `58.8` / `56.6`, and with the continuous cache pointer augmentation achieves perplexities of `53.5` / `53.0`.

First, train the model:

`python main.py --batch_size 20 --data data/penn --dropouti 0.4 --seed 28 --epoch 300 --save PTB.pt`

The first epoch should result in a validation perplexity of `308.03`.

To then fine-tune that model:

`python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --seed 28 --epoch 300 --save PTB.pt`

The validation perplexity after the first epoch should be `60.85`.

**Note:** Fine-tuning modifies the original saved model in `PTB.pt` - if you wish to keep the original weights you must copy the file.

Finally, to run the pointer:

`python pointer.py --data data/penn --save PTB.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000` 

Note that the model in the paper was trained for 500 epochs and the batch size was 40, in comparison to 300 and 20 for the model above.
The window size for this pointer is chosen to be 500 instead of 2000 as in the paper.

**Note:** BPTT just changes the length of the sequence pushed onto the GPU but won't impact the final result.

#### WikiText-2 (WT2)

The instruction below train a WT2 model that without finetuning achieves perplexities of `69.1` / `66.1` (validation / testing), with finetuning achieves perplexities of `68.7` / `65.8`, and with the continuous cache pointer augmentation achieves perplexities of `53.6` / `52.0` (`51.95` specifically).

`python main.py --seed 20923 --epochs 750 --data data/wikitext-2 --save WT2.pt`

The first epoch should result in a validation perplexity of `629.93`.

`python -u finetune.py --seed 1111 --epochs 750 --data data/wikitext-2 --save WT2.pt`

The validation perplexity after the first epoch should be `69.14`.

**Note:** Fine-tuning modifies the original saved model in `PTB.pt` - if you wish to keep the original weights you must copy the file.

Finally, run the pointer:

`python pointer.py --save WT2.pt --lambdasm 0.1279 --theta 0.662 --window 3785 --bptt 2000 --data data/wikitext-2`

**Note:** BPTT just changes the length of the sequence pushed onto the GPU but won't impact the final result.

## Speed

All the augmentations to the LSTM, including our variant of [DropConnect (Wan et al. 2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) termed weight dropping which adds recurrent dropout, allow for the use of NVIDIA's cuDNN LSTM implementation.
PyTorch will automatically use the cuDNN backend if run on CUDA with cuDNN installed.
This ensures the model is fast to train even when convergence may take many hundreds of epochs.

The default speeds for the model during training on an NVIDIA Quadro GP100:

+ Penn Treebank: approximately 45 seconds per epoch for batch size 40, approximately 65 seconds per epoch with batch size 20
+ WikiText-2: approximately 105 seconds per epoch for batch size 80

Speeds are approximately three times slower on a K80. On a K80 or other memory cards with less memory you may wish to enable [the cap on the maximum sampled sequence length](https://github.com/salesforce/awd-lstm-lm/blob/ef9369d277f8326b16a9f822adae8480b6d492d0/main.py#L131) to prevent out-of-memory (OOM) errors, especially for WikiText-2.

If speed is a major issue, SGD converges more quickly than our non-monotonically triggered variant of ASGD though achieves a worse overall perplexity.
