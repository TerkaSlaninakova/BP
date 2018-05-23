'''
File: generate.py
Author: Terézia Slanináková (xslani06), 
    repurposed from https://github.com/ibab/tensorflow-wavenet and https://github.com/tomlepaine/fast-wavenet

Based on trained model generates the resulting waveform through a generation process.
'''

import numpy as np
import tensorflow as tf
import librosa
import sys
from utils import *

def generate_sample(input, state, w, activation=None):
    '''
    Performs convolution for a single convolutional step.
    Slices the weight filter to only 1 current and 1 past value, multiplies if with input and state respectively.
    Args:
        input: single sapmle of current input
        state: past sample (recurrent state)
    Return:
        single generated output sample
    '''
    past_sample_w = w[0, :, :]
    current_sample_w = w[1, :, :]
    output = tf.matmul(input, current_sample_w) + tf.matmul(state, past_sample_w)
    if activation:
        output = activation(output)
    return output

class Generator():
    ''' Takes care of the whole generation procedure.'''
    def __init__(self, trainer, outdir, log):
        '''
        Defines variables used throughout the generation procedure.
        Args:
            trainer: Instance of the Trainer class, used to access properties of WaveNet
            outdir: Output directory to save results of generation to
            log: logger instance
        '''
        self.trainer = trainer
        self.outdir = outdir
        self.log = log
        self.sess = create_session()
        self.sess.run(tf.global_variables_initializer())
        self.current_sample = tf.placeholder(tf.int32)
        self.next_sample = self.generate_next_sample(self.current_sample)
        self.entropies = []
        self.entropies_to_display = []
        self.entropy_every = 50

    def generate(self, restore_from, n_samples, seed_from, teacher_forcing, sr, out_dir, log, should_plot_distr=False):
        '''
        Performs generation based on the generation scheme infered by the parameters
        Args:
            restore_from: Model to use for generation,not needed when running in the same process as training.
            n_samples: How many amplitudes to generate
            seed_from: Referential audio to use (the 'teacher')
            teacher_forcing: Decides whether teacher forcing should be used or not
            sr: sampling rate used
            out_dir: output directory to store plots in
            log: logger instance
            should_plot_distr: decides whether to plot gaussian distribution
        Return:
            Resulting waveform, loaded ground truth waveform and collected cross entropies
        '''
        if teacher_forcing and seed_from:
            return self.generate_teacher_forcing(restore_from, n_samples, seed_from, sr, should_plot_distr)
        elif seed_from:
            return self.generate_seed(restore_from, n_samples, seed_from, sr, should_plot_distr)
        else:
            return self.generate_unique(restore_from, n_samples, should_plot_distr)

    def generate_teacher_forcing(self, restore_from, n_samples, seed_from, tf_sr=8000, should_plot_distr=False):
        '''
        Performs the teacher forcing generation scheme
        Args:
            restore_from: Model to use for generation,not needed when running in the same process as training.
            n_samples: How many amplitudes to generate
            seed_from: Referential audio to use (the 'teacher')
            tf_sr: sampling rate used
            should_plot_distr: decides whether to plot gaussian distribution
        Return:
            Resulting waveform, loaded ground truth waveform and collected cross entropies
        '''
        cross_entropies = []
        cross_entropies_to_display = []
        self.sess.run(self.init_ops)
        if restore_from:
            load_weights(restore_from, self.sess, self.trainer.saver, self.log)
        self.log('Loaded ground truth audio from {}'.format(seed_from))
        gt_waveform, gt_waveform_bins = get_first_audio(seed_from, tf_sr)

        gt_waveform_bins = list(gt_waveform_bins)
        self.log('Starting teacher forcing generation with len(waveform_bins)={}'.format(len(gt_waveform_bins)))
        pred_waveform_bins = []
        n_samples = n_samples-1

        for step in range(n_samples):
            outputs = [self.next_sample]
            outputs.extend(self.push_ops)
            input = gt_waveform_bins[step]

            if step % 100 == 0:
                print(step,'/', n_samples)
                sys.stdout.flush()
            
            prediction = self.sess.run(outputs, feed_dict={self.current_sample: input})[0]
            pred_sample = np.argmax(prediction)
            pred_waveform_bins.append(pred_sample)
            self.entropies.append(entropy(prediction))
            cross_entropies.append(cross_entropy(prediction[gt_waveform_bins[step+1]])) 
            
            if step % 1000 == 0 and should_plot_distr:
                plot_gaussian_distr(self.outdir, 'pred_distr_' + str(step), prediction, pred_sample, gt_waveform_bins[step+1], True, self.log)
            if step % self.entropy_every == 0 and step != 0:
                self.entropies_to_display.append(np.mean(self.entropies[-self.entropy_every:]))
                cross_entropies_to_display.append(np.mean(cross_entropies[-self.entropy_every:]))
            
        out = mu_law_decode(np.array(pred_waveform_bins))
        out = np.insert(out, 0, gt_waveform[0], axis=0)

        return out

    def generate_seed(self, restore_from, n_samples, seed_from, sr=8000, should_plot_distr=False):
        '''
        Performs the seeded generation scheme
        Args:
            restore_from: Model to use for generation,not needed when running in the same process as training.
            n_samples: How many amplitudes to generate
            seed_from: Referential audio to use (the 'teacher')
            sr: sampling rate used
            should_plot_distr: decides whether to plot gaussian distribution
        Return:
            Resulting waveform
        '''
        self.sess.run(self.init_ops)
        if restore_from:
            load_weights(restore_from, self.sess, self.trainer.saver, self.log)
        waveform, waveform_bins = get_first_audio(seed_from, sr)
        waveform_bins = list(waveform_bins)
        self.log('Starting teacher forcing generation with len(waveform_bins)={}'.format(len(waveform_bins)))
        
        outputs = [self.next_sample]
        outputs.extend(self.push_ops)
        
        for sample in waveform_bins[-self.trainer.receptive_field: -1]:
            self.sess.run(outputs, feed_dict={self.current_sample: sample})
        
        for step in range(n_samples):
            outputs = [self.next_sample]
            outputs.extend(self.push_ops)
            input = waveform_bins[-1]

            if step % 100 == 0:
                print(step,'/', n_samples)
                sys.stdout.flush()
            prediction = self.sess.run(outputs, feed_dict={self.current_sample: input})[0]
            pred_sample = np.random.choice(np.arange(self.trainer.q_channels), p=prediction)
            self.entropies.append(entropy(prediction))
            if step % 1000 == 0 and should_plot_distr:
                plot_gaussian_distr(self.outdir, 'pred_distr_' + str(step), prediction, pred_sample, None, True, self.log)
            
            waveform_bins.append(pred_sample)
            if step % self.entropy_every == 0 and step != 0:
                self.entropies_to_display.append(np.mean(self.entropies[-self.entropy_every:]))

        out = mu_law_decode(np.array(waveform_bins[-n_samples:]))
        return out

    def generate_unique(self, restore_from, n_samples, should_plot_distr=False):
        '''
        Performs the unique generation scheme
        Args:
            restore_from: Model to use for generation,not needed when running in the same process as training.
            n_samples: How many amplitudes to generate
            should_plot_distr: decides whether to plot gaussian distribution
        Return:
            Resulting waveform
        '''
        self.sess.run(self.init_ops)
        if restore_from:
            load_weights(restore_from, self.sess, self.trainer.saver, self.log)
        waveform_bins = []
        waveform_bins.append(np.random.randint(self.trainer.q_channels))

        for step in range(n_samples):
            outputs = [self.next_sample]
            outputs.extend(self.push_ops)
            input = waveform_bins[-1]
            if step % 100 == 0:
                print(step,'/', n_samples)
                sys.stdout.flush()
            prediction = self.sess.run(outputs, feed_dict={self.current_sample: input})[0]
            pred_sample = np.random.choice(np.arange(self.trainer.q_channels), p=prediction)
            waveform_bins.append(pred_sample)
            self.entropies.append(entropy(prediction))
            if step % 1000 == 0 and should_plot_distr:
                plot_gaussian_distr(self.outdir, 'pred_distr_' + str(step), prediction, pred_sample, None, True, self.log)
            if step % self.entropy_every == 0 and step != 0:
                self.entropies_to_display.append(np.mean(self.entropies[-self.entropy_every:]))

        out = mu_law_decode(np.array(waveform_bins))
        return out

    def generate_dil(self, input, state, i):
        '''
        Generates output for one dilation layer.
        Collects the outputs for the filter and gate portion, performs activation with tanh and sigmoid
        and constructs the dense and skip output.
        Args:
            input: single sample of current input
            state: past sample (recurrent state)
            i: index of current dilation layer
        Return:
            skip and dense output
        '''
        current_dil_l_vars = self.trainer.net.variables['dil_stack'][i]
        output_kernel = generate_sample(input, state, current_dil_l_vars['kernel'])
        output_gate = generate_sample(input, state, current_dil_l_vars['gate'])
        
        out = tf.tanh(output_kernel) * tf.sigmoid(output_gate)
        dense = tf.matmul(out, current_dil_l_vars['dense'][0, :, :])
        skip = tf.matmul(out, current_dil_l_vars['skip'][0, :, :])

        return skip, input + dense


    def generate_next_sample(self, input_sample):
        '''
        Predicts the probability of the next sample based on current sample.
        Runs the raw input sample through the generation process, computes softmax probability on the output.
        Args:
            input_sample: current input sample
        Return:
            Probability distr. of the next sample.
        '''
        one_hot_input = tf.one_hot(input_sample, self.trainer.q_channels)
        one_hot_reshaped_input = tf.reshape(one_hot_input, [-1, self.trainer.q_channels])
        raw_output = self.construct_generator(one_hot_reshaped_input)
        generated_output = tf.reshape(raw_output, [-1, self.trainer.q_channels])
        probability = tf.nn.softmax(generated_output)
        last_sample = tf.slice(probability, [tf.shape(probability)[0] - 1, 0], [1, self.trainer.q_channels])
        return tf.reshape(last_sample, [-1])

    def construct_generator(self, one_hot_reshaped_input):
        '''
        The generation process, optimized with fast-wavenet.
        Iterated through the lyers of the network, sses caching into queues for storing immediate operations.
        See https://github.com/tomlepaine/fast-wavenet for more information.
        Args:
            one_hot_reshaped_input: a single one-hot reshaped input sample.
        Return:
            Probability distr. of the next sample.
        '''
        init_ops = []
        push_ops = []
        current_input = one_hot_reshaped_input
        convolution_queue = tf.FIFOQueue(1, dtypes=tf.float32, shapes=(1, self.trainer.q_channels))
        init = convolution_queue.enqueue_many(tf.zeros((1, 1, self.trainer.q_channels)))

        current_state = convolution_queue.dequeue()
        push = convolution_queue.enqueue([current_input])
        init_ops.append(init)
        push_ops.append(push)
        weights_filter = self.trainer.net.variables['causal_layer']['kernel']
        current_input = generate_sample(current_input, current_state, self.trainer.net.variables['causal_layer']['kernel'])
        outputs = []
        for i, dilation in enumerate(self.trainer.dilations):
            dilation_queue = tf.FIFOQueue(dilation, dtypes=tf.float32, shapes=(1, self.trainer.net.dil_w))
            init = dilation_queue.enqueue_many(tf.zeros((dilation, 1,  self.trainer.net.dil_w)))

            current_state = dilation_queue.dequeue()
            push = dilation_queue.enqueue([current_input])
            init_ops.append(init)
            push_ops.append(push)

            output, current_input = self.generate_dil(current_input, current_state, i)
            outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        variables = self.trainer.net.variables['pp']
        transformed = tf.nn.relu(sum(outputs))
        
        conv = tf.matmul(transformed, variables['pp1'][0, :, :])
        transformed = tf.nn.relu(conv)
        conv = tf.matmul(transformed, variables['pp2'][0, :, :])
        return conv
