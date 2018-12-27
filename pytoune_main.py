import logging
import argparse
import time
import math
import os
import hashlib

import numpy as np
import torch
import torch.nn as nn

import data
from data import SentenceLoader
import pytoune_model as m

from utils import batchify, get_batch, repackage_hidden
from splitcross import SplitCrossEntropyLoss

from pytoune.framework import Experiment as PytouneExperiment
from pytoune.framework.callbacks import ClipNorm, ReduceLROnPlateau
from callbacks import *


def get_source_directory(directory_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)


def get_experiment_directory(directory_name):
    default_dir = get_source_directory('./results')
    dest_directory = os.environ.get('RESULTS_DIR', default_dir)
    return os.path.join(dest_directory, directory_name)


def main():
    randomhash = ''.join(str(time.time()).split('.'))

    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers')
    parser.add_argument('--lr', type=float, default=30, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N', help='batch size')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--nonmono', type=int, default=5, help='random seed')
    parser.add_argument('--cuda', action='store_false', help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default=randomhash+'.pt', help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='', help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd', help='optimizer to use (sgd, adam)')
    parser.add_argument('--when', nargs="+", type=int, default=[-1], help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    args = parser.parse_args()
    args.tied = True

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    eval_batch_size = 20
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    train_loader = SentenceLoader(train_data, args.bptt)
    valid_loader = SentenceLoader(val_data, args.bptt)
    test_loader = SentenceLoader(test_data, args.bptt)

    ntokens = len(corpus.dictionary)
    model = m.RNNModel(
        args.model,
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayers,
        args.dropout,
        args.dropouth,
        args.dropouti,
        args.dropoute,
        args.wdrop,
        args.tied,
        args.alpha,
        args.beta
    )

    if args.model == 'QRNN': model.reset()

    ###
    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    device = None
    device_id = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id) # Fix bug where memory is allocated on GPU0 when ask to take GPU1.
        device = torch.device('cuda:%d' % device_id)
        logging.info("Training on GPU %d" % device_id)
    else:
        logging.info("Training on CPU")

    dataset = args.data.split('/')[-1]
    model_name = "AWD_{}_{}".format(args.model, dataset)
    expt_name = './expt_{}'.format(model_name)
    expt_dir = get_experiment_directory(expt_name)
    expt = PytouneExperiment(
        expt_dir,
        model,
        device=device,
        optimizer=optimizer,
        monitor_metric='val_loss',
        monitor_mode='min'
    )

    callbacks = [
        HiddenInitCallback(args.batch_size, eval_batch_size),
        HiddenRepackagingCallback(),
        AdaptativeLRSchedulerCallback(train_loader),
        ClipNorm(params, args.clip),
        EvaluationCallback(),
        ASGDOptimizerSwitchCallback(args),
        ReduceLROnPlateau(monitor='val_loss', mode='min', patience=20, factor=0.5, threshold_mode='abs', threshold=1e-3, verbose=True)
    ]

    expt.train(train_loader, valid_loader, callbacks=callbacks)
    expt.test(test_loader)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
