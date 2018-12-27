import torch
import logging
from pytoune.framework import Callback

class HiddenInitCallback(Callback):
    """
    This callback is used to reset the hidden state at each epoch.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs):
        self.model.model.init_hidden(self.batch_size)


class HiddenRepackagingCallback(Callback):
    """
    At each batch we need to repackage the hidden state
    so that the gradient does not backprop through the whole dataset.
    """
    def on_batch_begin(self, batch, logs):
        self.model.model.repackage_hidden()


class AdaptativeLRSchedulerCallback(Callback):
    """
    This callback computes a learning rate based on the sequence length and the bptt.
    """
    def __init__(self, loader):
        self.loader = loader
        self.lr2 = None

    def on_batch_begin(self, batch, logs):
        self.lr2 = self.model.optimizer.param_groups[0]['lr']
        effective_learning_rate = self.lr2 * self.loader.seq_len / self.loader.bptt
        self.model.optimizer.param_groups[0]['lr'] = effective_learning_rate

    def on_batch_end(self, batch, logs):
        self.model.optimizer.param_groups[0]['lr'] = self.lr2


class EvaluationCallback(Callback):
    """
    This callback copies some params before the epoch begins.
    It was not that clear why we did this in the original code, may need further documentation.
    """
    def __init__(self, ):
        self.tmp = {}

    def on_epoch_begin(self, epoch, logs):
        if 't0' in self.model.optimizer.param_groups[0]: # Check if we are in ASGD
            for prm in self.model.model.parameters():
                if prm in self.tmp:
                    prm.data = self.tmp[prm].clone()

    def on_epoch_end(self, epoch, logs):
        if 't0' in self.model.optimizer.param_groups[0]: # Check if we are in ASGD
            for prm in self.model.model.parameters():
                self.tmp[prm] = prm.data.clone()
                prm.data = self.model.optimizer.state[prm]['ax'].clone()


class ASGDOptimizerSwitchCallback(Callback):
    """
    This callback triggers the change of the SGD optimizer to ASGD,
    given we are using SGD (not Adam) and the validation loss is non mono.
    """
    def __init__(self, args):
        self.args = args
        self.val_losses = list()

    def can_switch_to_asgd(self):
        return self.args.optimizer == 'sgd' and 't0' not in self.model.optimizer.param_groups[0]

    def is_non_mono(self, val_loss):
        return len(self.val_losses) > self.args.nonmono and val_loss > min(self.val_losses[:-self.args.nonmono])

    def on_epoch_end(self, epoch, logs):
        val_loss = logs['val_loss']
        if self.can_switch_to_asgd() and self.is_non_mono(val_loss):
            logging.info("Switching to ASGD")
            self.model.optimizer = torch.optim.ASGD(
                self.model.model.parameters(),
                lr=self.args.lr,
                t0=0,
                lambd=0.,
                weight_decay=self.args.wdecay
            )
        self.val_losses.append(val_loss)


class MetricsCallback(Callback):
    """
    This callback logs a bunch of information about parameters, gradients etc.
    As suggested by https://github.com/allenai/writing-code-for-nlp-research-emnlp2018/blob/master/writing_code_for_nlp_research.pdf
    """
    def __init__(self, logger):
        super(MetricsCallback, self).__init__()
        self.logger = logger

    def on_backward_end(self, batch):
        for parameter, values in self.model.model.named_parameters():
            self.logger.log_scalar("{}.grad.mean".format(parameter), float(values.mean()))
            self.logger.log_scalar("{}.grad.std".format(parameter), float(values.std()))

    def on_batch_end(self, batch, logs):
        self.logger.log_scalar("steps.train.loss", logs['loss'])
        self.logger.log_scalar("steps.train.acc", logs['acc'])

    def on_epoch_end(self, epoch, logs):
        self.logger.log_scalar("epochs.train.loss", logs['loss'])
        self.logger.log_scalar("epochs.train.acc", logs['acc'])
        self.logger.log_scalar("epochs.val.loss", logs['val_loss'])
        self.logger.log_scalar("epochs.val.acc", logs['val_acc'])
