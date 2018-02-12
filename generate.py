import numpy as np
import tensorflow as tf
import librosa
import argparse

from utils import plot, timestamp

def causal(inputs, state, name=None, activation=None):
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('w')
        output = tf.matmul(inputs, w[1, :, :]) + tf.matmul(state, w[0, :, :])
        if activation:
            output = activation(output)
    return output

def output(layer, name=''):
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('w')[0, :, :]
        b = tf.get_variable('b')

        output = tf.matmul(layer, w) + tf.expand_dims(b, 0)
    return output

class Generator:
    def __init__(self, model):
        self.model = model
        inputs = tf.placeholder(tf.float32, [1, 1], name='sample')
        current_output = inputs

        init_conv_queues = []
        init_recurrent_states = []
        state_size = 1
        for block in range(self.model.n_blocks):
            for dilation in self.model.dilations:
                name = 'block{}-dil{}'.format(block, dilation)

                q = tf.FIFOQueue(dilation, dtypes=tf.float32, shapes=(1, state_size))
                # initialize all recurrent states
                init = q.enqueue_many(tf.zeros((dilation, 1, state_size)))
                # pop first recurrent state
                state_ = q.dequeue()
                rec_state = q.enqueue([current_output])
                init_recurrent_states.append(rec_state)
                init_conv_queues.append(init)
                # computes new output and recurrent states
                current_output = causal(current_output, state_, name=name, activation=tf.nn.relu)
                if state_size == 1:
                    state_size = self.model.n_hidden

        outputs = output(current_output)
        outputs = [tf.nn.softmax(outputs)]
        outputs.extend(init_recurrent_states)

        self.inputs = inputs
        self.outputs = outputs
        self.init_conv_queues = init_conv_queues
        
        self.model.sess.run(self.init_conv_queues)

    def run(self, input, num_samples):
        predictions = []
        template = np.linspace(-1, 1, self.model.n_classes)
        for step in range(num_samples):
            output = self.model.sess.run(self.outputs, feed_dict={self.inputs: input})[0]
            value = np.argmax(output[0, :])

            input = np.array(template[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                print(step, '/', num_samples)

        preditions_ = np.concatenate(predictions, axis=1)
        return preditions_[0]