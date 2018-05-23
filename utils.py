'''
File: utils.py
Author: Terézia Slanináková (xslani06)

Collection of various functions for supporting operations (plotting, loading/preparing dataset, saving audio)
'''
import datetime
import time
import librosa
from matplotlib import pyplot as plt
import math
import numpy as np
import librosa.display
import os
import tensorflow as tf
import glob
import sys
from random import shuffle
import GPUtil
import resource
import scipy.stats as stats

# Setups for font size adjustment
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

plt.switch_backend('agg')
CURRENT_RUN_TIMESTAMP = None

def str2bool(v):
    '''Parses the boolean command-line arguments.'''
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_output_dir(output_dir):
	''' Creates output directory '''
	return os.path.dirname(os.path.realpath(__file__)) + '/' + timestamp() + '/' if output_dir == '' else output_dir

def prepare_datasets(directory, log, train_val_ratio = 5):
	'''
	Localized and splits the dataset to validation and training batches.
	The files are
	Args:
		directory: location of the dataset
		log: logging instance
		train_val_ratio: ratio with which dataset should be split (80:20 by default)
	Return:
		List of filenames of training and validation data. 
	'''
	if directory[-1] != '/':
		directory += '/'
	files = list(glob.iglob(directory + '*wav'))
	log('Found {} files in {}'.format(len(files), directory))
	how_many_val_files = len(files) // train_val_ratio
	if how_many_val_files == 0:
		print('Couldnt create and appropriate validation dataset, there is too few samples in the dataset: ', len(files), ' exiting')
		exit()
	log('Assigning {} audios to be validation data'.format(how_many_val_files))
	val_dataset = [file for i, file in enumerate(files) if i % train_val_ratio == 0]
	train_dataset = [file for i, file in enumerate(files) if i % train_val_ratio != 0]
	return train_dataset, val_dataset

def mu_law_encode(signal, quantization_channels=256):
	'''
	Performs inverse mu-law encoding on an audio
	Args:
		signal: incoming encoded audio signal
		quantization_channels: number of quant. levels
	Return:
		Decoded signal.
	'''
	mu = quantization_channels - 1
	magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
	signal = np.sign(signal) * magnitude
	signal = (signal + 1) / 2 * mu + 0.5
	return signal.astype(np.int32)

def mu_law_decode(signal, quantization_channels=256):
	'''
	Performs mu-law encoding on an audio
	Args:
		signal: incoming audio signal
		quantization_channels: number of quant. levels
	Return:
		Encoded signal.
	'''
	mu = quantization_channels - 1
	y = signal.astype(np.float32)
	y = 2 * (y / mu) - 1
	x = np.sign(y) * (1.0 / mu) * ((1.0 + mu)**abs(y) - 1.0)
	return x

def get_first_audio(path, sr=8000):
	'''
	Loads a single audio. Used in generation to get referential audio.
	Args:
		path: path the the audio file
		sr: used sampling rate
	Return:
		List of loaded audios as numpy arrays, encoded audio
	'''
	audio, _ = librosa.load(path, sr=sr, mono=True)
	audio = audio.reshape(-1, 1).T[0].T
	return audio, mu_law_encode(audio)

def create_audio(filenames, sr=8000):
	'''
	Loads the audio dataset.
	Args:
		filenames: list of filenames to load
		sr: used sampling rate
	Return:
		List of loaded audios as numpy arrays
	'''
	data = []
	audios = []
	for filename in filenames:
		print('loading ', filename)
		audio, _ = librosa.load(filename, sr=sr, mono=True)
		audio = audio.reshape(-1, 1).T[0].T
		audios.append(audio[:, None])
	return audios

def write_data(outdir, name, data, sr, log):
	'''
	Saves the resulting audio.
	Args:
		outdir: directory to save to
		name: name of the plot
		data: list of amplitudes
		sr: used sampling rate
		log: logger instance 
	'''
	create_out_dir(outdir, log)
	data = np.array(data)
	log('Saving generated wav as {}'.format(outdir + name))
	librosa.output.write_wav(outdir + name, data, sr)

def timestamp():
	''' Makes a unique timestamp for the whole run '''
	global CURRENT_RUN_TIMESTAMP
	if not CURRENT_RUN_TIMESTAMP:
		CURRENT_RUN_TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
	return CURRENT_RUN_TIMESTAMP

def cross_entropy(value, q_channels=256):
	''' Calculates the cross entropy, used as qualitative marker in generation process.'''
	return -np.log(value)

def entropy(value):
	''' Calculates the entropy, used as qualitative marker in generation process.'''
	return stats.entropy(value)

class Log:
	''' Logger takes care of customized logging messages throughout the run if logger should be used. '''
	def __init__(self,should_log):
		self.should_log = should_log
	
	def log(self, name, object_to_log=None):
		if self.should_log:
			if object_to_log is not None and LOG:
				print("[D] {}: {} {}".format(name,object_to_log.dtype,object_to_log.get_shape()))
			else:
				print("[D] {}".format(name))

def create_session(process_fraction=0.85):
	''' 
	Creates the GPU session with max. 1 gpu.
		Args:
			process_fraction: How many percent of GPU's power shuold be used.
		Return:
			Initialized session ready to be used by TF
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=process_fraction)
	config = tf.ConfigProto(
		gpu_options=gpu_options,
		intra_op_parallelism_threads=1, 
		inter_op_parallelism_threads=1,
		device_count={'GPU': 1})
	sess = tf.Session(config=config)
	return sess

def prepare_environment(resource_limit, log):
	'''	Prepares the environment by choosing one GPU to run on, 
		adjusts CUDA_VISIBLE_DEVICES env var for TF,
		sets the max. process length so that the training won't time out.'''
	try:
		DEVICE_ID_LIST = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.85, verbose=True)
		DEVICE_ID = DEVICE_ID_LIST[0]
		os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
		log('Preparing environment by choosing a gpu {} and setting resource limit={}'.format(DEVICE_ID, resource_limit))
	except:
		print('No GPU found, continuing in CPU mode.')
	try:
		soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
		resource.setrlimit(resource.RLIMIT_CPU, (resource_limit, hard))
	except:
		print('No limit set.')

def save_weights(saver, outdir, epoch, iteration, sess, loss, log):
	'''
	Saves model's weights, used in training to save the best model.
	Args:
		saver: TF Saver instance
		outdir: directory to save to
		epoch: which epoch is the training at
		iteration: which iteration is the training at
		sess: TF Session instance to run the saving
		loss: currently achieved loss
		log: logger instance
	'''
	create_out_dir(outdir, log)
	checkpoint_dir = outdir + 'saved_weights/'
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_path = checkpoint_dir + timestamp() + '_epoch' + epoch + '_loss=' + loss + '_model.ckpt'
	log('Storing checkpoint as {} ...'.format(checkpoint_path))
	saver.save(sess, checkpoint_path, global_step=iteration, write_meta_graph=False)

def load_weights(load_dir, sess, saver, log):
	'''
	Loads model's weights, to used in training .
	Args:
		load_dir: Where to load from
		sess: TF Session instance to run the loading
		saver: Saver TF instance for saving and loading
		log: logger instance
	Returns:
		0 if viable model was not found, 
		information about the state of the loaded model otherwise
	'''
	checkpoint = tf.train.get_checkpoint_state(load_dir)
	if checkpoint:
		print("Checkpoint: ", checkpoint.model_checkpoint_path)
		step = int(checkpoint.model_checkpoint_path.split('-')[-1])
		last_loss = float(checkpoint.model_checkpoint_path.split('=')[1].split('_')[0])
		if checkpoint.model_checkpoint_path.split('epoch')[-2] != checkpoint.model_checkpoint_path:
			last_epoch = int(checkpoint.model_checkpoint_path.split('epoch')[-1].split('_')[0])
		else:
			last_epoch = 0
		saver.restore(sess, checkpoint.model_checkpoint_path)
		return step, last_loss, last_epoch
	return 0, None, None

def plot_gaussian_distr(outdir, name, prediction, chosen_sample, should_plot, log):
	'''
	Plots gaussian distribution. Used to visualize softmax' predictions of the output.
	Args:
		outdir: directory to save to
		name: name of the plot
		prediction: the prob. dirstribution of predictions 
		chosen_sample: one value from prob. distr. chosen by the generation process
		should_plot: decides, if plotting should be used at all
		log: logger instance 
	'''
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(10, 4))
		plt.plot(np.arange(256), prediction)
		plt.scatter(chosen_sample, prediction[chosen_sample], color=['green'])
		log('Saving plot of the gaussian distribution as \'{}\''.format(outdir + name))
		plt.xlabel('bin')
		plt.ylabel('probability')
		plt.savefig(outdir + name, dpi=100)

def plot_waveform(outdir, name, data, sr, should_plot, log):
	'''
	Plots the resulting waveform.
	Args:
		outdir: directory to save to
		name: name of the plot
		data: list of amplitudes
		sr: used sampling rate
		should_plot: decides, if plotting should be used at all
		log: logger instance 
	'''
	if should_plot:
		create_out_dir(outdir, log)
		times = np.arange(len(data))/float(sr)
		fig = plt.figure(figsize=(60, 8))
		host = fig.add_subplot(111)
		plt.plot(times, data)
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name, dpi=100)

def plot_spectogram(outdir, name, data, sr, should_plot, log):
	'''
	Plots spectogram. Used as a qualitative measure for the resuling audio.
	Args:
		outdir: directory to save to
		name: name of the plot
		data: list of amplitudes
		sr: used sampling rate
		should_plot: decides, if plotting should be used at all
		log: logger instance 
	'''
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(30, 10))
		D = librosa.amplitude_to_db(librosa.core.magphase(librosa.stft(data))[0])
		librosa.display.specshow(D, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Linear-frequency power spectrogram')
		log('Saving spectogram as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def plot_entropy(outdir, name, entropies, should_plot, log):
	'''
	Plots entropies. Used as a qualitative measure for the resuling audio.
	Args:
		outdir: directory to save to
		name: name of the plot
		entropies: list of entropies
		should_plot: decides, if plotting should be used at all
		log: logger instance 
	'''
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(24, 12))
		range_entropies = [i for i in range(len(entropies))]
		plt.title('Entropy of probabilities')
		plt.xlabel('Sample')
		plt.ylabel('Entropy')
		plt.plot(range_entropies, entropies)
		plt.grid(True)
		log('Saving entropy plot as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def plot_losses(outdir, name, losses, losses_val, loss_every, epoch_every, start_at, should_plot, log):
	'''
	Plots the training process.
	Args:
		outdir: directory to save to
		name: name of the plot
		losses: list of training losses
		losses_val: list of validation losses
		loss_every: how many iterations apart was a validation loss registered
		epoch_every: how many iterations is one epoch (number of training samples)
		start_at: what iteration did the training start at
		should_plot: decides, if plotting should be used at all
		log: logger instance 
	'''
	if should_plot:
		create_out_dir(outdir, log)
		fig = plt.figure(figsize=(12, 6))
		ax1 = fig.add_subplot(111)
		iterations_range_losses = [(i*loss_every+start_at) for i in range(len(losses))]
		iterations_range_val_losses = [(i*epoch_every+start_at) for i in range(len(losses_val))]
		ax1.plot(iterations_range_losses, losses, label='training loss')
		ax1.plot(iterations_range_val_losses, losses_val, label='valid. loss')
		plt.legend(loc='upper right')
		plt.xlabel('epochs')
		plt.ylabel('losses')
		plt.title('Training process')
		plt.grid(True)
		log('Saving plot of losses as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def create_out_dir(path, log):
	''' Creates a dedicated directory for the run if one does not exist already.'''
	if path is None or not os.path.exists(path):
		log('Creating directory for storing data of the run: \'{}\''.format(path))
		os.makedirs(path)