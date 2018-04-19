import datetime
import time
import librosa
from matplotlib import pyplot as plt
import math
import numpy as np
import librosa.display
import os
import tensorflow as tf
from scipy.io import wavfile
import glob
import sys
from random import shuffle
import GPUtil
import resource
import matplotlib.mlab as mlab
import scipy.stats as stats

plt.switch_backend('agg')
CURRENT_RUN_TIMESTAMP = None

def prepare_datasets(directory, log):
	if directory[-1] != '/':
		directory += '/'
	files = list(glob.iglob(directory + '*wav'))
	log('Found {} files in {}'.format(len(files), directory))
	shuffle(files)
	how_many_val_files = len(files) // 10
	if how_many_val_files == 0:
		print('Couldnt create and appropriate validation dataset, there is too few samples in the dataset: ', len(files), ' exiting')
		exit()
	log('Assigning {} to be validation data'.format(how_many_val_files))
	val_dataset = files[:how_many_val_files]
	train_dataset = files[how_many_val_files:]
	return train_dataset, val_dataset

def create_logspace_template(n_channels=256):
	pos = list(np.logspace(-10, 0.00000001, n_channels/2, base=math.e).reshape(-1, 1).T[0])
	neg = list(reversed(-np.logspace(-10, 0.00000001, n_channels/2, base=math.e).reshape(-1, 1).T[0]))
	template = np.array(neg + pos)
	return template

def get_first_audio(path, sr=8000, template=create_logspace_template()):
	audio, _ = librosa.load(path, sr=sr, mono=True)
	audio = audio.reshape(-1, 1).T[0].T
	if len(audio) > 72000:
		audio = audio[:72000]
	bins = np.digitize(audio, template) - 1

	return template[bins], bins

def create_audio(filenames, sample_rate=8000, template=create_logspace_template()):
	data = []
	for filename in filenames:
		print('loading ', filename)
		audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
		#rate, data = wavfile.read(filename)
		#print(rate);print(data);exit()
		audio = audio.reshape(-1, 1).T[0].T
		if len(audio) > 120000:
			audio = audio[:120000]
		if template is not None:
			bins = np.digitize(audio, template) - 1
			data.append(template[bins][:, None])
	#print(data.shape);print(audio[:, None].shape);exit()
	return data

def write_data(outdir, name, data, output_sample_rate, log):
	create_out_dir(outdir, log)
	data = np.array(data)
	log('Saving generated wav as {}'.format(outdir + name))
	librosa.output.write_wav(outdir + name, data, output_sample_rate)

def timestamp():
	# ensure only 1 unique timestamp per run
	global CURRENT_RUN_TIMESTAMP
	if not CURRENT_RUN_TIMESTAMP:
		CURRENT_RUN_TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
	return CURRENT_RUN_TIMESTAMP

def cross_entropy(A, Y):
	return -(1.0/256) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

class Log:
	def __init__(self,should_log):
		self.should_log = should_log
	
	def log(self, name, object_to_log=None):
		if self.should_log:
			if object_to_log is not None and LOG:
				print("[D] {}: {} {}".format(name,object_to_log.dtype,object_to_log.get_shape()))
			else:
				print("[D] {}".format(name))

def create_session():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
	config = tf.ConfigProto(
		gpu_options=gpu_options,
		intra_op_parallelism_threads=1, 
		inter_op_parallelism_threads=1,
		device_count={'GPU': 1})
	sess = tf.Session(config=config)
	return sess

def prepare_environment(resource_limit, log):
	DEVICE_ID_LIST = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.1, verbose=True)
	DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

	os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
	log('Preparing environment by choosing a gpu {} and setting resource limit={}'.format(DEVICE_ID, resource_limit))
	soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
	resource.setrlimit(resource.RLIMIT_CPU, (resource_limit, hard))

def plot_gaussian_distr(outdir, name, data, chosen_sample, gt, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		fig = plt.figure(figsize=(10, 4))
		host = fig.add_subplot(111)
		par = host.twinx()
		host.plot(np.arange(256), data)
		par.scatter(chosen_sample, data[chosen_sample], color=['green'])
		if gt:
			par.scatter(gt, data[gt], color=['red'])
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		host.set_xlabel('bin')
		host.set_ylabel('probability')
		plt.savefig(outdir + name, dpi=100)

def plot_waveform(outdir, name, data, sr, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		times = np.arange(len(data))/float(sr)
		fig = plt.figure(figsize=(60, 4))
		host = fig.add_subplot(111)
		plt.plot(times, data)
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name, dpi=100)

def plot_two_waveforms(outdir, name, gt, data, sr, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		times = np.arange(len(data))/float(sr)
		fig = plt.figure(figsize=(80, 4))
		ax1 = fig.add_subplot(111)
		ax1.plot(times,data, label='predictions') 
		ax1.plot(times,gt, label='ground truth') 
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		plt.legend(loc='upper right')
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name, dpi=100)


def plot_three_waveforms(outdir, name, gt, data, random_data, sr, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		times = np.arange(len(data))/float(sr)
		fig = plt.figure(figsize=(60, 4))
		ax1 = fig.add_subplot(111)
		ax1.plot(times,random_data, label='predictions (random)') 
		ax1.plot(times,data, label='predictions (argmax)') 
		ax1.plot(times,gt, label='ground truth') 
		plt.legend(loc='upper right')
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name, dpi=100)

def plot_spectogram(outdir, name, data, sr, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(8, 4))
		D = librosa.amplitude_to_db(librosa.core.magphase(librosa.stft(data))[0])
		librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		#plt.title('Linear power spectrogram (grayscale)')
		log('Saving spectogram as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def plot_entropy(outdir, name, entropies, spacing_int, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(12, 4))
		range_entropies = [i*spacing_int for i in range(len(entropies))]
		plt.title('Entropy of probabilities')
		plt.xlabel('Sample')
		plt.ylabel('Entropy')
		plt.plot(range_entropies, entropies)
		plt.grid(True)
		log('Saving entropy plot as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def plot_two_entropies(outdir, name, entropies_1, entropies_2, spacing_int, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		fig = plt.figure(figsize=(24, 12))
		ax1 = fig.add_subplot(111)
		range_entropies = [i*spacing_int for i in range(len(entropies_1))]
		plt.title('Entropy of probabilities')
		plt.xlabel('Sample')
		plt.ylabel('Entropy')
		ax1.plot(range_entropies, entropies_1, label='Entropy')
		ax1.plot(range_entropies, entropies_2,  label='Cross-entropy')
		plt.legend(loc='upper right')
		plt.grid(True)
		log('Saving entropy plot as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)


def plot_losses(outdir, name, losses, losses_val, loss_every, val_every, epoch_every, epochs, start_at, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		fig = plt.figure(figsize=(12, 6))
		ax1 = fig.add_subplot(111)
		if epochs != 0:
			iterations_range_losses = [(i*loss_every+start_at)/epoch_every for i in range(len(losses))]
			iterations_range_val_losses = [(i*epoch_every+start_at)/epoch_every for i in range(len(losses_val))]
		else:
			iterations_range_losses = [(i*loss_every+start_at) for i in range(len(losses))]
			iterations_range_val_losses = [(i*epoch_every+start_at) for i in range(len(losses_val))]
		print(iterations_range_losses)
		print(iterations_range_val_losses)
		#epoch_range = [i for i in range(epochs+1)]
		#print(epoch_range);exit()
		#plt.scatter(iterations_range, losses)
		ax1.plot(iterations_range_losses, losses, label='training loss')#, s=10, c='b', marker="s", label='training loss')
		ax1.plot(iterations_range_val_losses, losses_val, label='valid. loss')#, s=10, c='r', marker="o", label='valid. loss')
		#ax1.set_xticks(epoch_range)
		plt.legend(loc='upper right');
		plt.xlabel('epochs')
		plt.ylabel('losses')
		plt.title('Training process')
		plt.grid(True)
		log('Saving plot of losses as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def create_out_dir(path, log):
	if not os.path.exists(path):
		log('Creating directory for storing data of the run: \'{}\''.format(path))
		os.makedirs(path)