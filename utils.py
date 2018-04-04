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

plt.switch_backend('agg')
CURRENT_RUN_TIMESTAMP = None

def prepare_datasets(directory, log):
	#return list(glob.iglob(directory + 'vio/' + '*wav')) + list(glob.iglob(directory + 'pia/' + '*wav')) + list(glob.iglob(directory + 'gac/' + '*wav'))
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

def get_first_audio(path, template=create_logspace_template()):
	audio, _ = librosa.load(path, sr=8000, mono=True)
	audio = audio.reshape(-1, 1).T[0].T
	if len(audio) > 72000:
		audio = audio[:72000]
	bins = np.digitize(audio, template) - 1

	return template[bins][:, None], bins

def create_audio(filenames, sample_rate, template=create_logspace_template()):
	for filename in filenames:
		audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
		audio = audio.reshape(-1, 1).T[0].T
		if len(audio) > 120000:
			audio = audio[:120000]
		if template is not None:
			bins = np.digitize(audio, template) - 1
			yield template[bins][:, None]
		else:
			yield audio

class Reader():
	def __init__(self, train_data, validation_data, coord, val_coord, sample_rate, receptive_field):
		self.train_data = train_data
		self.validation_data = validation_data
		self.sample_rate = sample_rate
		self.coord = coord
		self.val_coord = val_coord
		self.receptive_field = receptive_field
		self.audio = tf.placeholder(dtype=tf.float32, shape=None)
		self.queue = tf.PaddingFIFOQueue(32, ['float32'], shapes=[(None, 1)])
		self.enqueue = self.queue.enqueue([self.audio])
		self.sample_size = None

	def enqueue_train_audio(self, sess, should_stop):
		enqueue_audio(self, sess, should_stop, self.coord, self.train_data)

	def enqueue_val_audio(self, sess, should_stop):
		enqueue_audio(self, sess, should_stop, self.val_coord, self.validation_data)

	def enqueue_audio(self, sess, should_stop, coord, dataset):
		while not should_stop:
			iterator = create_audio(dataset, self.sample_rate)
			for audio in iterator:
				if coord.should_stop():
					should_stop = True; break
				audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]], mode='constant')
				sess.run(self.enqueue, feed_dict={self.audio: audio})

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

def plot_waveform(outdir, name, data, sr, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		times = np.arange(len(data))/float(sr)
		plt.figure(figsize=(30, 4))
		plt.fill_between(times,data) 
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
		plt.title('Linear power spectrogram (grayscale)')
		log('Saving spectogram as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def plot_losses(outdir, name, losses, losses_val, val_every, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		fig = plt.figure(figsize=(12, 6))
		ax1 = fig.add_subplot(111)
		iterations_range_losses = [i for i in range(len(losses))]
		iterations_range_val_losses = [i*val_every if i!=0 else 0 for i in range(len(losses_val))]
		#plt.scatter(iterations_range, losses)
		ax1.plot(iterations_range_losses, losses, label='training loss')#, s=10, c='b', marker="s", label='training loss')
		ax1.plot(iterations_range_val_losses, losses_val, label='valid. loss')#, s=10, c='r', marker="o", label='valid. loss')
		plt.legend(loc='upper right');
		plt.xlabel('iterations')
		plt.ylabel('losses')
		plt.title('Training process')
		plt.grid(True)
		log('Saving plot of losses as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def create_out_dir(path, log):
	if not os.path.exists(path):
		log('Creating directory for storing data of the run: \'{}\''.format(path))
		os.makedirs(path)