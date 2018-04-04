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
    def __init__(self, train_data, validation_data, q_channels, dil_channels, skip_channels, n_dilations, n_blocks, gpu_fraction, learning_rate, sample_rate, kernel_w, log):
        self.q_channels = q_channels
        self.dilations = n_blocks*[2**x for x in range(n_dilations)]
        self.receptive_field = (kernel_w - 1) * sum(self.dilations) + kernel_w
        self.dilation_channels = dil_channels
        # Register audio input reader, that will be launched during training
        self.coordinator = tf.train.Coordinator()
        self.val_coordinator = tf.train.Coordinator()
        self.reader = Reader(train_data, validation_data, self.coordinator, self.val_coordinator, sample_rate=sample_rate, receptive_field=self.receptive_field)

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

    def _save_weights(self, outdir, iteration, sess, loss, log):
        create_out_dir(outdir, log)
        checkpoint_dir = outdir + 'saved_weights/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + timestamp() + '_loss=' + loss + '_model.ckpt'
        log('Storing checkpoint as {} ...'.format(checkpoint_path))
        self.saver.save(sess, checkpoint_path, global_step=iteration, write_meta_graph=False)

    def _load_weights(self, load_dir, sess, log):
        checkpoint = tf.train.get_checkpoint_state(load_dir)
        if checkpoint:
            print("Checkpoint: ", checkpoint.model_checkpoint_path)
            step = int(checkpoint.model_checkpoint_path.split('-')[-1])
            last_loss = float(checkpoint.model_checkpoint_path.split('=')[1].split('_')[0])
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            return step, last_loss
        return 0, None

    def train(self, target_loss, train_iterations, output_dir, should_plot, load_dir, log):
        losses = []
        saved_model_losses = []
        val_losses = []
        stop_iteration = train_iterations
        sess = create_session()
        sess.run(tf.global_variables_initializer())
       
        init_step = 0
        if load_dir:
            init_step, last_loss = self._load_weights(load_dir, sess, log)
            saved_model_losses.append(last_loss)
            log('Last loss was: {}'.format(last_loss))

        print("Starting training, initial step: ", init_step)
        log_every = 10
        assert log_every > 0
        #save_every = train_iterations // 1000
        
        loss = None
        val_every = 200
        assert val_every > 0
        # initialize input data supply though threads
        threads = tf.train.start_queue_runners(sess=sess, coord=self.coordinator)
        thread = threading.Thread(target=self.reader.enqueue_train_audio, args=(sess, False))
        thread.daemon = True
        thread.start()
        try:
            for iter in range(init_step, train_iterations):
                stop_iteration = iter
                loss, _ = sess.run([self.loss, self.optim]) 

                if len(saved_model_losses) == 0:
                    self._save_weights(output_dir, iter, sess, str(loss), log)
                    saved_model_losses.append(loss)
                    print('Stored loss {}'.format(loss));sys.stdout.flush()

                if iter % log_every == 0:
                    print(iter, '/', train_iterations, ': ', loss)
                    sys.stdout.flush()

                losses.append(loss)
                if iter % val_every == 0 and iter != 0:
                    threads_ = tf.train.start_queue_runners(sess=sess, coord=self.val_coordinator)
                    thread = threading.Thread(target=self.reader.enqueue_val_audio, args=(sess, False))
                    thread.daemon = True
                    thread.start()
                    loss_ = sess.run([self.loss])
                    loss_ = loss_[0]
                    self.val_coordinator.request_stop()
                    self.val_coordinator.join(threads_)
                    print('Validation loss: ', loss_);sys.stdout.flush()
                    val_losses.append(loss_)
                    if loss_ < saved_model_losses[len(saved_model_losses)-1]:
                        self._save_weights(output_dir, iter, sess, str(loss_), log)
                        saved_model_losses.append(loss_)   
                        print('Stored loss {}'.format(loss_))
                        sys.stdout.flush()                

        except (KeyboardInterrupt,Exception):
            pass
        finally:
            if len(saved_model_losses) > 0:
                print("Final loss: ", saved_model_losses[len(saved_model_losses)-1])
            else:
                print("No saved loss")
            self.coordinator.request_stop()
            self.coordinator.join(threads)
            sys.stdout.flush()
        
        plot_losses(output_dir, 'training_process_' + timestamp() + '.png', losses, val_losses, val_every, should_plot, log)
        return losses