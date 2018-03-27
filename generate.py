import numpy as np
import tensorflow as tf
import librosa
import sys
from utils import create_session

def mu_law_decode(output, q_channels):
    '''
    Performs inverse operation of the u-law encoding to recover the waveform
    '''
    mu = q_channels - 1
    signal = 2 * (tf.to_float(output) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude

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
    def __init__(self, trainer):
        self.trainer = trainer
        self.sess = create_session()
        self.sess.run(tf.global_variables_initializer())
        
    def generate(self, restore_from, n_samples, log):
        current_sample = tf.placeholder(tf.int32)

        next_sample = self.generate_next_sample(current_sample)
        self.sess.run(self.init_ops)

        if restore_from:
            self.trainer._load_weights(restore_from, self.sess, log)

        waveform = []
        # start the generation with one random sample
        waveform.append(np.random.randint(self.trainer.q_channels))
        for step in range(n_samples):
            outputs = [next_sample]
            outputs.extend(self.push_ops)
            window = waveform[-1]
            if step % 1000 == 0:
                print(step,'/', n_samples)
                sys.stdout.flush()
            prediction = self.sess.run(outputs, feed_dict={current_sample: window})[0]
            # based on predicted set of samples chose one with conditional probability
            sample = np.random.choice(np.arange(self.trainer.q_channels), p=prediction)
            waveform.append(sample)
        decode = mu_law_decode(current_sample, self.trainer.q_channels)
        out = self.sess.run(decode, feed_dict={current_sample: waveform})
        return out

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
            dilation_queue = tf.FIFOQueue(dilation, dtypes=tf.float32, shapes=(1, self.trainer.dilation_channels))
            init = dilation_queue.enqueue_many(tf.zeros((dilation, 1, self.trainer.dilation_channels)))

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
