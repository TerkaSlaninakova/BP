import numpy as np
import tensorflow as tf
import librosa
import argparse

def generate_sample(inputs, state, name=None, activation=None):
    '''Performs convolution for a single convolutional step'''
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('w')
        past_sample_w = w[0, :, :]
        current_sample_w = w[1, :, :]
        output = tf.matmul(inputs, current_sample_w) + tf.matmul(state, past_sample_w)
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
        self.inputs = tf.placeholder(tf.float32, [1, 1], name='sample')
        current_sample = self.inputs

        init_ops = []
        push_ops = []
        state_size = 1
        for block in range(self.model.n_blocks):
            for dilation in self.model.dilations:
                name = 'block{}-dil{}'.format(block, dilation)

                convolution_queue = tf.FIFOQueue(dilation, dtypes=tf.float32, shapes=(1, state_size))
                # initializing convolution queues by setting all of their recurrent states to zeros
                init = convolution_queue.enqueue_many(tf.zeros((dilation, 1, state_size)))

                # pop first recurrent state, feed it to corresp. location
                current_state = convolution_queue.dequeue()

                push = convolution_queue.enqueue([current_sample])
                init_ops.append(init)
                push_ops.append(push)
                # computes new output and recurrent states
                current_sample = generate_sample(current_sample, current_state, name=name, activation=tf.nn.relu)

                if state_size == 1:
                    state_size = self.model.n_hidden

        outputs = output(current_sample)
        outputs = [tf.nn.softmax(outputs)]
        outputs.extend(push_ops)

        self.outputs = outputs
        self.init_ops = init_ops
        
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples, restore_from):
        predictions = []
        if restore_from:
            self.model.load_weights(restore_from)

        template = np.linspace(-1, 1, self.model.n_classes)
        for step in range(num_samples):
            output = self.model.sess.run(self.outputs, feed_dict={self.inputs: input})[0]
            value = np.random.choice(np.arange(self.model.n_classes), p=output[0, :])

            input = np.array(template[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                print(step, '/', num_samples)

        preditions_ = np.concatenate(predictions, axis=1)
        return preditions_[0]