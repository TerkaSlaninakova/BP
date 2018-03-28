import os
from utils import create_out_dir, timestamp, create_session, Reader, create_audio, plot_losses
import tensorflow as tf
from wavenet import Wavenet
import sys
import time
import threading

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
    def __init__(self, data_dir, q_channels, dil_channels, skip_channels, n_dilations, n_blocks, gpu_fraction, learning_rate, sample_rate, kernel_w, log):
        self.q_channels = q_channels
        self.dilations = n_blocks*[2**x for x in range(n_dilations)]
        self.receptive_field = (kernel_w - 1) * sum(self.dilations) + kernel_w
        self.dilation_channels = dil_channels
        # Register audio input reader, that will be launched during training
        self.coordinator = tf.train.Coordinator()
        self.reader = Reader(data_dir, self.coordinator, sample_rate=sample_rate, receptive_field=self.receptive_field)

        net = Wavenet(dilations=self.dilations, kernel_width=kernel_w, dilation_channels=dil_channels, skip_channels=skip_channels, q_channels=self.q_channels, receptive_field=self.receptive_field, log=log)

        input_batch = self.reader.queue.dequeue_many(1)
        one_hot = one_hot_mu_law_encode(input_batch, self.q_channels)
        raw_output = net.construct_network(one_hot)
        target_output = construct_target_output(one_hot, self.q_channels, self.receptive_field)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(raw_output, [-1, self.q_channels]), labels=target_output)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainable = tf.trainable_variables()
        self.loss = tf.reduce_mean(loss)
        self.optim = optimizer.minimize(self.loss, var_list=trainable)
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.net = net

    def _save_weights(self, outdir, iteration, sess, log):
        sys.stdout.flush()
        create_out_dir(outdir, log)
        checkpoint_dir = outdir + 'saved_weights/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + timestamp() + '_model.ckpt'
        print('Storing checkpoint as {} ...'.format(checkpoint_path), end="")
        self.saver.save(sess, checkpoint_path, global_step=iteration)

    def _load_weights(self, load_dir, sess, log):
        checkpoint = tf.train.get_checkpoint_state(load_dir)
        if checkpoint:
            print("Checkpoint: ", checkpoint.model_checkpoint_path)
            step = int(checkpoint.model_checkpoint_path.split('-')[-1])
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            return step
        return 0

    def train(self, target_loss, train_iterations, output_dir, should_plot, load_dir, log):
        losses = []
        saved_model_losses = []
        stop_iteration = train_iterations
        sess = create_session()
        sess.run(tf.global_variables_initializer())
       
        init_step = 0
        if load_dir:
            init_step = self._load_weights(load_dir, sess, log)
        
        print("Starting training, initial step: ", init_step)
        log_every = train_iterations // 10000
        save_every = train_iterations // 1000
        
        loss = None
        # initialize input data supply though threads
        threads = tf.train.start_queue_runners(sess=sess, coord=self.coordinator)
        threading.Thread(target=self.reader.enqueue_audio, args=(sess, False)).start()
        try:
            for iter in range(init_step, train_iterations):
                stop_iteration = iter
                loss, _ = sess.run([self.loss, self.optim])
                if iter % save_every == 0:
                    if len(saved_model_losses) > 0 and loss < saved_model_losses[len(saved_model_losses)-1]:
                        self._save_weights(output_dir, iter, sess, log)
                        saved_model_losses.append(loss)
                    elif len(saved_model_losses) == 0:
                        self._save_weights(output_dir, iter, sess, log)
                        saved_model_losses.append(loss)
                if iter % log_every == 0:
                    print(iter, '/', train_iterations, ': ', loss)
                    sys.stdout.flush()
                
                if loss < target_loss:
                    break
                losses.append(loss)
        except Exception:
            pass
        finally:
            #if stop_iteration % save_every != 0:
            #    self._save_weights(output_dir, stop_iteration, sess, log)
            if len(saved_model_losses) > 0:
                print("Final loss: ", saved_model_losses[len(saved_model_losses)-1])
            else:
                print("No saved loss")
            self.coordinator.request_stop()
            self.coordinator.join(threads)
            sys.stdout.flush()
        plot_losses(output_dir, 'training_process_' + timestamp() + '.png', losses, len(losses), should_plot, log)
        return losses