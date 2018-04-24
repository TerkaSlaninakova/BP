import numpy as np
import tensorflow as tf
import librosa
import sys
from utils import *
import scipy.stats

def generate_sample(inputs, state, w, activation=None):
    '''
    Performs convolution for a single convolutional step
    '''
    past_sample_w = w[0, :, :]
    current_sample_w = w[1, :, :]
    output = tf.matmul(inputs, current_sample_w) + tf.matmul(state, past_sample_w)
    if activation:
        output = activation(output)
    return output

class Generator():
    def __init__(self, trainer, outdir, log):
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

    def generate_teacher_forcing(self, restore_from, n_samples, seed_from, tf_sr=8000, should_plot_distr=False):
        cross_entropies = []
        cross_entropies_to_display = []
        self.sess.run(self.init_ops)
        if restore_from:
            self.trainer._load_weights(restore_from, self.sess, self.log)
        gt_waveform, gt_waveform_bins = get_first_audio(seed_from, tf_sr)
        gt_waveform_bins = list(gt_waveform_bins)
        print('Starting TF generation with len(waveform_bins)=', len(gt_waveform_bins))
        pred_waveform_bins = []
        pred_waveform_bins_random = []
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
            pred_sample_random = np.random.choice(np.arange(self.trainer.q_channels), p=prediction)
            pred_waveform_bins_random.append(pred_sample_random)
            self.entropies.append(entropy(prediction))
            cross_entropies.append(cross_entropy(prediction[gt_waveform_bins[step+1]])) 
            if step % 1000 == 0 and should_plot_distr:
                plot_gaussian_distr(self.outdir, 'pred_distr_' + str(step), prediction, pred_sample, gt_waveform_bins[step+1], True, self.log)
            if step % self.entropy_every == 0 and step != 0:
                self.entropies_to_display.append(np.mean(self.entropies[-self.entropy_every:]))
                cross_entropies_to_display.append(np.mean(cross_entropies[-self.entropy_every:]))
            
        out = mu_law_decode(np.array(pred_waveform_bins))
        out = np.insert(out, 0, gt_waveform[0], axis=0)
        out_rand = mu_law_decode(np.array(pred_waveform_bins_random[-n_samples:]))
        out_rand = np.insert(out, 0, gt_waveform[0], axis=0)
        return out, out_rand, gt_waveform, cross_entropies_to_display

    def generate_seed(self, restore_from, n_samples, seed_from, sr=8000, should_plot_distr=False):
        self.sess.run(self.init_ops)
        if restore_from:
            self.trainer._load_weights(restore_from, self.sess, self.log)
        waveform, waveform_bins = get_first_audio(seed_from, sr)
        waveform_bins = list(waveform_bins)
        print('Starting seeding generation with len(waveform_bins)=', len(waveform_bins))
        # priming the generation with seed
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
            #pred_sample = np.argmax(prediction)
            self.entropies.append(entropy(prediction))
            if step % 1000 == 0 and should_plot_distr:
                #pred_sample = np.random.choice(np.arange(self.trainer.q_channels), p=prediction)
                plot_gaussian_distr(self.outdir, 'pred_distr_' + str(step), prediction, pred_sample, None, True, self.log)
            waveform_bins.append(pred_sample)
            if step % self.entropy_every == 0 and step != 0:
                self.entropies_to_display.append(np.mean(self.entropies[-self.entropy_every:]))

        out = mu_law_decode(np.array(waveform_bins[-n_samples:]))
        out_seed = mu_law_decode(np.array(waveform_bins[self.trainer.receptive_field:]))
        # out now contains bins from seed (out[:n_samples]) as well as generated (out[n_samples:])
        #plot_waveform(self.outdir, 'waveforms_' + timestamp() + '.png', out_seed[-100:], out[:100], n_samples, n_samples, True, self.log)
        #exit()
        return out

    def generate_unique(self, restore_from, n_samples, should_plot_distr=False):
        self.sess.run(self.init_ops)
        if restore_from:
            self.trainer._load_weights(restore_from, self.sess, self.log)
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
            #pred_sample = np.argmax(prediction)
            waveform_bins.append(pred_sample)
            self.entropies.append(entropy(prediction))
            if step % 1000 == 0 and should_plot_distr:
                plot_gaussian_distr(self.outdir, 'pred_distr_' + str(step), prediction, pred_sample, None, True, self.log)
            if step % self.entropy_every == 0 and step != 0:
                self.entropies_to_display.append(np.mean(self.entropies[-self.entropy_every:]))

        out = mu_law_decode(np.array(waveform_bins))
        return out
    '''
    def generate(self, n_samplese):

            if seed_from and teacher_forcing:
                waveform_.append(sample)
                waveform_pred_.append(sample_)
            else:
                waveform.append(sample)
                preds.append(sample)
        #decode = mu_law_decode(current_sample, self.trainer.q_channels)

        if seed_from and teacher_forcing:
            out = mu_law_decode(np.array(waveform_[-n_samples:]))
            #print(out);print(len(out));print(type(out));print(type(out[0]))
            #new_o.append(waveform_fl)
            out = np.insert(out, 0, waveform_fl[0], axis=0)
            #out = [waveform_fl[0]] + out
            #print(out);print(len(out));exit()
            out_rand = mu_law_decode(np.array(waveform_pred_[-n_samples:]))
            out_rand = [waveform_fl[0]] + out_rand
        else:
            out = mu_law_decode(np.array(waveform))
#def plot_waveform(outdir, name, data, sr, should_plot, log):
        
        plot_waveform(outdir, 'waveforms_' + timestamp() + '.png', out[:n_samples], out[n_samples:], n_samples, len(out), True, log)
        #exit()
        if teacher_forcing and seed_from:
            #print('here');print(waveform_fl[:10])
            #def plot_waveform(outdir, name, data, data2, div, sr, should_plot, log):

            plot_waveform(outdir, 'waveforms_' + timestamp() + '.png', out[:n_samples], out[n_samples:], n_samples, len(out), True, log)
            exit()
            plot_two_entropies(outdir, 'entropies_' + timestamp() + '.png', entropies_to_display, cross_entropies_to_display, entropy_every, True, log)
            plot_two_waveforms(outdir, 'waveforms_'+timestamp()+'.png', waveform_fl[:n_samples], out[:n_samples], n_samples, True, log)
            plot_three_waveforms(outdir, 'waveforms_'+timestamp()+'.png', waveform_fl[:1000], out[:1000], out_rand[:1000], n_samples, True, log)
        return out[:n_samples]
    '''

    def generate_dil(self, input, state, i):
        current_dil_l_vars = self.trainer.net.variables['dil_stack'][i]
        output_kernel = generate_sample(input, state, current_dil_l_vars['kernel'])
        output_gate = generate_sample(input, state, current_dil_l_vars['gate'])
        
        out = tf.tanh(output_kernel) * tf.sigmoid(output_gate)
        dense = tf.matmul(out, current_dil_l_vars['dense'][0, :, :])
        skip = tf.matmul(out, current_dil_l_vars['skip'][0, :, :])

        return skip, input + dense


    def generate_next_sample(self, input_sample):
        '''
        Predicts the probability of the next sample based on current sample
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
        Generates a waveform of samples based on input
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
