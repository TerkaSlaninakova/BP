import os
from utils import create_out_dir, timestamp, create_session, create_audio, plot_losses
import tensorflow as tf
from wavenet import Wavenet
import sys
import time
import threading
import numpy as np
from random import shuffle

def one_hot_mu_law_encode(audio, q_channels):
    '''
    Performs u-law encoding and one-hot encodes the input audio
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
    target_output = tf.slice(tf.reshape(encoded_audio, [1, -1, q_channels]), [0, receptive_field, 0], [-1, -1, -1])
    target_output = tf.reshape(target_output, [-1, q_channels])
    return target_output

class Trainer():
    def __init__(self, q_channels, dil_width, res_width, skip_width, kernel_width, n_dilations, n_blocks, learning_rate, log):
        self.q_channels = q_channels
        self.dilations = n_blocks*[2**x for x in range(n_dilations)]
        self.receptive_field = (kernel_width - 1) * sum(self.dilations) + kernel_width
        net = Wavenet(dilations=self.dilations, kernel_width=kernel_width, dilation_width=dil_width, residual_width=res_width, skip_width=skip_width, 
            q_channels=self.q_channels, receptive_field=self.receptive_field, log=log)

        input_batch = tf.placeholder(dtype=tf.float32, shape=None)
        one_hot = one_hot_mu_law_encode(input_batch, self.q_channels)
        raw_output = net.construct_network(one_hot)
        target_output = construct_target_output(one_hot, self.q_channels, self.receptive_field)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(raw_output, [-1, self.q_channels]), labels=target_output)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainable = tf.trainable_variables()
        self.loss = tf.reduce_mean(loss)
        self.optim = optimizer.minimize(self.loss, var_list=trainable)
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.input_batch = input_batch
        self.net = net

    def _save_weights(self, outdir, epoch, iteration, sess, loss, log):
        create_out_dir(outdir, log)
        checkpoint_dir = outdir + 'saved_weights/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + timestamp() + '_epoch' + epoch + '_loss=' + loss + '_model.ckpt'
        log('Storing checkpoint as {} ...'.format(checkpoint_path))
        self.saver.save(sess, checkpoint_path, global_step=iteration, write_meta_graph=False)

    def _load_weights(self, load_dir, sess, log):
        checkpoint = tf.train.get_checkpoint_state(load_dir)
        if checkpoint:
            print("Checkpoint: ", checkpoint.model_checkpoint_path)
            step = int(checkpoint.model_checkpoint_path.split('-')[-1])
            last_loss = float(checkpoint.model_checkpoint_path.split('=')[1].split('_')[0])
            if checkpoint.model_checkpoint_path.split('epoch')[-2] != checkpoint.model_checkpoint_path:
                last_epoch = int(checkpoint.model_checkpoint_path.split('epoch')[-1].split('_')[0])
            else:
                last_epoch = 0
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            return step, last_loss, last_epoch
        return 0, None, None

    def train(self, train_batch, val_batch, target_loss, train_epochs, output_dir, should_plot, load_dir, log, log_every=50):
        assert log_every > 0
        losses = []
        saved_model_losses = []
        val_losses = []
        sess = create_session()
        sess.run(tf.global_variables_initializer())
        last_epoch = 0
        start_at = 0
        init_step = 0
        if load_dir:
            init_step, last_loss, last_epoch = self._load_weights(load_dir, sess, log)
            saved_model_losses.append(last_loss)
            log('Last loss was: {}'.format(last_loss))
        if last_epoch != 0:
            start_at = last_epoch
        print("Starting training, initial step: ", init_step)
        start = time.time()
        loss = None
        losses_to_display = []
        epoch_counter = last_epoch
        epoch_length = len(train_batch)
        try:
            for iter in range(init_step, epoch_length*train_epochs):
                if iter%epoch_length == 0 and iter != 0:
                    shuffle(train_batch)
                    shuffle(val_batch)
                    epoch_counter += 1

                loss, _ = sess.run([self.loss, self.optim], feed_dict={self.input_batch: train_batch[iter%epoch_length], self.net.keep_prob : 1.0}) 
                losses.append(loss)

                if iter % log_every == 0:
                    print(epoch_counter, '/', iter, '/', train_epochs, ': ', loss)
                    sys.stdout.flush()
                    losses_to_display.append(np.mean(losses[-log_every:]))

                if iter % epoch_length == 0:
                    loss_ = sess.run([self.loss], feed_dict={self.input_batch: val_batch[iter%len(val_batch)], self.net.keep_prob : 1.0})[0]
                    print('Validation loss: ', loss_);sys.stdout.flush()
                    val_losses.append(loss_)
                    if len(saved_model_losses) == 0 or len(saved_model_losses) > 0 and loss_ < saved_model_losses[len(saved_model_losses)-1]:
                        self._save_weights(output_dir, str(epoch_counter), iter, sess, str(loss_), log)
                        saved_model_losses.append(loss_)
                        print('Stored loss {}'.format(loss_));sys.stdout.flush()

        except KeyboardInterrupt:
            pass

        finally:
            if len(saved_model_losses) > 0:
                print("Final loss: ", saved_model_losses[len(saved_model_losses)-1])
            else:
                print("No saved loss")
            sys.stdout.flush()
        end = time.time()
        '''
        loss_ = sess.run([self.loss], feed_dict={self.input_batch: val_batch[iter%len(val_batch)], self.net.keep_prob : 1.0})[0]
        print('Validation loss: ', loss_);sys.stdout.flush()
        val_losses.append(loss_)
        if len(val_losses) == 0 or len(val_losses) > 0 and loss_ < val_losses[len(val_losses)-1]:
            self._save_weights(output_dir, str(epoch_counter), iter, sess, str(loss_), log)
            saved_model_losses.append(loss_)
            print('Stored loss {}'.format(loss_));sys.stdout.flush()
        losses_to_display.append(np.mean(losses[-log_every:]))
        '''
        plot_losses(output_dir, 'training_process_' + timestamp() + '.png', losses_to_display, val_losses, log_every, epoch_length, epoch_length, epoch_counter, start_at, should_plot, log)
        f = open(output_dir + 'training_losses', 'w')
        f.write('\n'.join(map(str, losses_to_display)))
        f.close()
        f = open(output_dir + 'val_losses', 'w')
        f.write('\n'.join(map(str, val_losses)))
        f.close()
        print('Training took ', end-start, ' s')
        return losses