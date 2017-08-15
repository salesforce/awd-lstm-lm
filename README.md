# AWD-LSTM Language Model

### Averaged Stochastic Gradient Descent with Weight Dropped LSTM

To recreate results in paper:

+ Train the base model using `main.py`
+ Finetune the model using `finetune.py`
+ Apply the continuous cache pointer to the finetuned model using `pointer.py`

## Reproduction

PTB (approx): `python main.py --seed 1111 --dropouti 0.4 --data data/penn --batch_size 40`
WikiText-2 (exact - valid ppl @ epoch 1 = 629.93): `python main.py --seed 20923`

The command for training the WikiText-2 (WT2) model should give you exact results equal to those reported in [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182).
The command for training the Penn Treebank (PTB) model should give approximately equivalent results to those reported in the paper, with the exact reproduction being lost due to small changes in the code.

## Speed

+ Penn Treebank takes approximately 50 seconds per epoch (NVIDIA Quadro GP100)
+ WikiText-2 takes 105 seconds per epoch (NVIDIA Quadro GP100)
