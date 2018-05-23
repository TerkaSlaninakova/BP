'''
File: train.py
Author: Terézia Slanináková (xslani06), repurposed from https://github.com/ibab/tensorflow-wavenet

Initializes the model using wavenet.py, pre-processes audio input, runs the training process.
'''

import os
from utils import *
import tensorflow as tf
from wavenet import Wavenet
import sys
import time
import threading
import numpy as np
from random import shuffle

def one_hot_mu_law_encode(audio, q_channels=256):
    '''
    Performs u-law encoding and one-hot encodes the input audio.
    Args:
        audio - placeholder for the input to pre-process
        q_channels - number of quantization levels
    Return:
        pre-processed input
    '''
    mu = tf.to_float(q_channels - 1)
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
    signal = tf.sign(audio) * magnitude

    # Signal is quantized to q_channels
    mu_law_encoded = tf.to_int32((signal + 1) / 2 * mu + 0.5)
    one_hot = tf.one_hot(mu_law_encoded, depth=q_channels, dtype=tf.float32)
    one_hot = tf.reshape(one_hot, [1, -1, q_channels])
    return one_hot

def construct_target_output(encoded_audio, q_channels, receptive_field):
    ''' 
    Creates the 'targets' from the pre-pr. input, that will be used in supervised training in logits/labels situation.
    Slices the input to receptive field size, reshapes to q_channels to be used by the network.
    Args:
        encoded_audio: input audio to make targets from
        q_channels: number of quant. levels used
        receptive_field: rec. field length
    Return:
        Targets ready to be used in training.
    '''
    target_output = tf.slice(tf.reshape(encoded_audio, [1, -1, q_channels]), [0, receptive_field, 0], [-1, -1, -1])
    target_output = tf.reshape(target_output, [-1, q_channels])
    return target_output

class Trainer():
    ''' Takes care of the whole training procedure, with inputs and targets initialization.'''
    def __init__(self, q_channels, dil_width, res_width, skip_width, kernel_width, n_dilations, n_blocks, learning_rate, log):
        '''
        Pre-processes audio, calls Wavenet to create the networks, creates the targets,
        prepares the loss calculation, optimizer and weights saver
        Args:
            q_channels: number of quantization levels
            dil_width:  width of dilation convolutional filters
            res_width: width of residual convolutional filters
            skip_width: width of skip channels for the softmax output
            kernel_width: filter with, first dimension of of kernel and gate dilation component
            n_dilations: Number of dilations to be used in every block
            n_blocks: Number of blocks in the network
            learning_rate: Learning rate to be used
            log: callback to log method
        '''
        self.q_channels = q_channels
        self.dilations = n_blocks*[2**x for x in range(n_dilations)]
        self.receptive_field = (kernel_width - 1) * sum(self.dilations) + kernel_width
        net = Wavenet(dilations=self.dilations, 
                      kernel_width=kernel_width, 
                      dilation_width=dil_width, 
                      residual_width=res_width, 
                      skip_width=skip_width, 
                      q_channels=self.q_channels, 
                      receptive_field=self.receptive_field, 
                      log=log)

        input_batch = tf.placeholder(dtype=tf.float32, shape=None)
        one_hot = one_hot_mu_law_encode(input_batch, self.q_channels)
        log('Constructed wavenet.')
        raw_output = net.construct_network(one_hot)
        target_output = construct_target_output(one_hot, self.q_channels, self.receptive_field)
        log('Constructed targets.')
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(raw_output, [-1, self.q_channels]), labels=target_output)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        log('Created optimizer.')
        trainable = tf.trainable_variables()
        self.loss = tf.reduce_mean(loss)
        self.optim = optimizer.minimize(self.loss, var_list=trainable)
        self.saver = tf.train.Saver(var_list=trainable)
        self.input_batch = input_batch
        self.net = net

    def train(self, train_batch, val_batch, train_epochs, output_dir, should_plot, load_dir, log, dropout=1.0, log_every=50):
        '''
        The training process.
        Runs the training procedure until train_epochs is met, saves the best model continuously based
        achieved on validation loss. The training process is timed.
        Args:
            train_batch: loaded training batch
            val_batch: loaded validation batch
            train_epochs: how many epochs to train for
            output_dir: where to save the data about training to 
            should_plot: decides whether to plot losses graph or not
            load_dir: directory to load model from, if there is any
            log: callback to log method
            dropout: the percentage of neurons, that should be kept during training. 1.0, meaning dropout of 0% by default.
            log_every: how often should the training process message be logged
        '''
        assert log_every > 0
        # lists to save losses to
        losses = []
        losses_to_display = []
        saved_model_losses = []
        val_losses = []

        sess = create_session()
        sess.run(tf.global_variables_initializer())

        last_epoch = 0
        init_step = 0
        if load_dir:
            init_step, last_loss, last_epoch = _load_weights(load_dir, sess, log)
            saved_model_losses.append(last_loss)
            log('Last loss was: {}'.format(last_loss))
        print("Starting training, initial step: ", init_step)
        start = time.time()
        loss = None
        epoch_counter = last_epoch
        epoch_length = len(train_batch)
        try:
            for iter in range(init_step, epoch_length*train_epochs):
                if iter%epoch_length == 0 and iter != 0:
                    shuffle(train_batch)
                    epoch_counter += 1

                loss, _ = sess.run(
                    [self.loss, self.optim], 
                    feed_dict={self.input_batch: train_batch[iter%epoch_length], 
                               self.net.keep_prob : 1.0}) 
                losses.append(loss)

                if iter % log_every == 0:
                    print(epoch_counter, '/', iter, '/', train_epochs, ': ', loss)
                    losses_to_display.append(np.mean(losses[-log_every:]))

                if iter % epoch_length == 0:
                    loss_ = sess.run(
                        [self.loss], 
                        feed_dict={self.input_batch: val_batch[iter%len(val_batch)], 
                                   self.net.keep_prob : dropout})[0]
                    print('Validation loss: ', loss_)
                    val_losses.append(loss_)
                    if len(saved_model_losses) == 0 or len(saved_model_losses) > 0 \
                            and loss_ < saved_model_losses[len(saved_model_losses)-1]:
                        save_weights(self.saver, output_dir, str(epoch_counter), iter, sess, str(loss_), log)
                        saved_model_losses.append(loss_)
                        print('Stored loss {}'.format(loss_))

        except KeyboardInterrupt:
            pass

        finally:
            if len(saved_model_losses) > 0:
                print("Final loss: ", saved_model_losses[len(saved_model_losses)-1])
            else:
                print("No saved loss")
            sys.stdout.flush()
        end = time.time()
        plot_losses(output_dir, 'training_process_' + timestamp() + '.png', losses_to_display, val_losses, \
            log_every, epoch_length, epoch_length, epoch_counter, init_step, should_plot, log)
        print('Training took ', end-start, ' s')
        return losses