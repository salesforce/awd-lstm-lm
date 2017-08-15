# AWD-LSTM Language Model

### Averaged Stochastic Gradient Descent with Weight Dropped LSTM

The codebase presented allows reproduction of the results from [Salesforce Research](https://einstein.ai/)'s [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) paper.
The model comes with instructions to train a word level language model over the Penn Treebank (PTB) and [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2) datasets, though the model is likely extensible to many other datasets.

+ Train the base model using `main.py`
+ Finetune the model using `finetune.py`
+ Apply the continuous cache pointer to the finetuned model using `pointer.py`

## Reproduction

#### Penn Treebank (PTB)

`python main.py --seed 1111 --dropouti 0.4 --data data/penn --batch_size 40 --epochs 500`

The codebase was modified after the paper in the model was produced and the random seeds differ, preventing exact reproduction, though the overall result should be similar however.
Note that the model in the paper was trained only for 500 epochs.

#### WikiText-2 (WT2)

`python main.py --seed 20923 --epochs 750`

This should reproduce the model from the paper exactly and can be confirmed by seeing the validation perplexity at epoch 1 hitting `629.93`.

## Speed

The augmentations to the LSTM, including our variant of [DropConnect (Wan et al. 2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) termed weight dropping, allow for the use of NVIDIA's cuDNN LSTM implementation.
This ensures the model is fast to train even when convergence may take many hundreds of epochs.

The default speeds for the model during training on an NVIDIA Quadro GP100 running on:

+ Penn Treebank: approximately 50 seconds per epoch
+ WikiText-2: 105 seconds per epoch

If speed is a major issue, SGD converges more quickly than the non-monotonically triggered variant of ASGD, though achieves a worse overall perplexity.
