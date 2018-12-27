import math
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_batch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class SentenceLoader:
    def __init__(self, dataset, bptt, train_mode=True):
        self.bptt = bptt
        self.seq_len = bptt
        self.dataset = dataset
        self.train_mode = train_mode

    def __iter__(self):
        i = 0
        while i < self.dataset.size(0) - 1 - 1:
            if self.train_mode:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                self.seq_len = max(5, int(np.random.normal(bptt, 5)))
            batch = get_batch(self.dataset, i, bptt, self.seq_len)
            i += self.seq_len
            yield batch

    def __len__(self):
        return len(self.dataset) // self.bptt
