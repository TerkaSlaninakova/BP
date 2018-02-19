import numpy as np
import tensorflow as tf
import argparse
from utils import plot_costs, timestamp
import os

NUM_OF_PERMITTED_THREADS = 1

def pad_inputs(inputs, dilation, width, n_channels):
    width_pad = int(dilation * np.ceil((width + dilation) / dilation))
    pad = width_pad - width

    shape = [int(width_pad / dilation), -1, n_channels]
    padded = tf.pad(inputs, [[0, 0], [pad, 0], [0, 0]])
    transposed = tf.transpose(padded, perm=[1, 0, 2])
    reshaped = tf.reshape(transposed, shape)
    outputs = tf.transpose(reshaped, perm=[1, 0, 2])
    return outputs

def pad_outputs(inputs, rate, crop_left=0):
    shape = tf.shape(inputs)    
    out_width = tf.to_int32(shape[1] * rate)

    _, _, num_channels = inputs.get_shape().as_list()

    transposed = tf.transpose(inputs, perm=[1, 0, 2])    
    reshaped = tf.reshape(transposed, [out_width, -1, num_channels])
    outputs = tf.transpose(reshaped, perm=[1, 0, 2])
    cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
    return cropped

def create_convolution_layer(inputs, out_channels, filter_width=2, output=False):
	input_channels = inputs.get_shape().as_list()[-1]
	if not output:
		stddev = np.sqrt(2)/np.sqrt(4*input_channels)
	else:
		stddev = 1/np.sqrt(2)
	w_init = tf.random_normal_initializer(stddev=stddev)
	w = tf.get_variable(name='w', shape=(filter_width, input_channels, out_channels), initializer=w_init)
	outputs = tf.nn.conv1d(inputs, w, stride=1, padding='VALID',data_format='NHWC')

	if output:
		b = tf.get_variable(name='b',shape=(out_channels, ),initializer=tf.constant_initializer(0.0))
		outputs = outputs + tf.expand_dims(tf.expand_dims(b,0), 0)
	else:
		outputs = tf.nn.relu(outputs)
	return outputs


def create_dilated_convolution_layer(inputs, out_channels, dilation=1,name=None):
	with tf.variable_scope(name):
		_, width, n_channels = inputs.get_shape().as_list()
		padded_input = pad_inputs(inputs, dilation=dilation, width=width, n_channels=n_channels)
		convolution_outputs = create_convolution_layer(padded_input, out_channels=out_channels)
		_, conv_out_width, _ = convolution_outputs.get_shape().as_list()
		new_width = conv_out_width * dilation

		outputs = pad_outputs(convolution_outputs, rate=dilation, crop_left=new_width-width)

		#reshape to [None, width == input_width, number_of_hidden_layers]
		tensor_shape = [tf.Dimension(None),tf.Dimension(width),tf.Dimension(out_channels)]
		outputs.set_shape(tf.TensorShape(tensor_shape))
		return outputs

class Net:
	def __init__(self, n_samples, n_classes, n_dilations, n_blocks, n_hidden, gpu_fraction):
		self.n_samples = n_samples
		self.dilations = [2**x for x in range(n_dilations)]
		self.n_channels = 1
		self.n_classes = n_classes
		self.n_blocks = n_blocks
		self.n_hidden = n_hidden
		inputs = tf.placeholder(tf.float32, shape=(None, self.n_samples, self.n_channels))
		targets = tf.placeholder(tf.int32, shape=(None, self.n_samples))
		
		layer = inputs
		for block in range(n_blocks):
			for dilation in self.dilations:
				layer = create_dilated_convolution_layer(layer, self.n_hidden, dilation=dilation, name='block{}-dil{}'.format(block, dilation))

		outputs = create_convolution_layer(layer, self.n_classes,filter_width=1,output=True)
		costs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
		cost = tf.reduce_mean(costs)
		
		train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
		config = tf.ConfigProto(
			gpu_options=gpu_options,
			intra_op_parallelism_threads=NUM_OF_PERMITTED_THREADS, 
			inter_op_parallelism_threads=NUM_OF_PERMITTED_THREADS,
			device_count={'GPU': 1})
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())

		self.inputs = inputs
		self.targets = targets
		self.sess = sess
		self.cost = cost
		self.train_step = train_step

	def _save_weights(self, output_dir, iteration):
		saver = tf.train.Saver(var_list=tf.trainable_variables())
		checkpoint_dir = output_dir + timestamp() + '/'
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		checkpoint_path = checkpoint_dir + timestamp() + '_model.ckpt'
		print('Storing checkpoint as {} ...'.format(checkpoint_path), end="")
		saver.save(self.sess, checkpoint_path, global_step=iteration)

	def load_weights(self, load_dir):
		checkpoint = tf.train.get_checkpoint_state(load_dir)
		if checkpoint:
			saver = tf.train.Saver(var_list=tf.trainable_variables())
			print("Checkpoint: ", checkpoint.model_checkpoint_path)
			step = int(checkpoint.model_checkpoint_path.split('-')[-1])
			saver.restore(self.sess, checkpoint.model_checkpoint_path)
			return step
		return 0

	def train(self, inputs, targets, target_cost, train_iterations, output_dir, should_plot, should_save_weights, load_dir=None):
		costs = []
		target_cost = target_cost
		stop_iteration = train_iterations
		init_step = 0
		if load_dir:
			init_step = self.load_weights(load_dir)
		print("init step: ", init_step)
		for iter in range(init_step, train_iterations):
			try:
				feed_dict = {self.inputs: inputs, self.targets: targets}
				cost, _ = self.sess.run([self.cost, self.train_step], feed_dict=feed_dict)
				if iter%50 == 0:
					print(iter,'/',train_iterations, ': ', cost)
				if cost < target_cost:
					stop_iteration = iter
					break
				costs.append(cost)
			except KeyboardInterrupt:
				stop_iteration = iter
				break
		if should_save_weights:
			self._save_weights(output_dir, stop_iteration)
		plot_costs('training_process_' + timestamp() + '.png', costs, len(costs), should_plot)
		return