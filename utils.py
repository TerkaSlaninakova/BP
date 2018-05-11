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
from scipy.special import xlogy

font = {'family' : 'calibri',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

plt.switch_backend('agg')
CURRENT_RUN_TIMESTAMP = None

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_output_dir(output_dir):
	os.path.dirname(os.path.realpath(__file__)) + '/' + timestamp() + '/' if output_dir == '' else output_dir

def prepare_datasets(directory, log, train_val_ratio = 5):
	if directory[-1] != '/':
		directory += '/'
	files = list(glob.iglob(directory + '*wav'))
	log('Found {} files in {}'.format(len(files), directory))
	shuffle(files)
	how_many_val_files = len(files) // train_val_ratio
	if how_many_val_files == 0:
		print('Couldnt create and appropriate validation dataset, there is too few samples in the dataset: ', len(files), ' exiting')
		exit()
	log('Assigning {} audios to be validation data'.format(how_many_val_files))
	val_dataset = [file for i, file in enumerate(files) if i % train_val_ratio == 0]
	train_dataset = [file for i, file in enumerate(files) if i % train_val_ratio != 0]
	return train_dataset, val_dataset

def create_logspace_template(n_channels=256):
	pos = list(np.logspace(-10, 0.00000001, n_channels/2, base=math.e).reshape(-1, 1).T[0])
	neg = list(reversed(-np.logspace(-10, 0.00000001, n_channels/2, base=math.e).reshape(-1, 1).T[0]))
	template = np.array(neg + pos)
	return template

def mu_law_encode(signal, quantization_channels=256):
    mu = quantization_channels - 1
    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude
    signal = (signal + 1) / 2 * mu + 0.5
    return signal.astype(np.int32)

def mu_law_decode(signal, quantization_channels=256):
    mu = quantization_channels - 1
    y = signal.astype(np.float32)
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu)**abs(y) - 1.0)
    return x

def get_first_audio(path, sr=8000):
	audio, _ = librosa.load(path, sr=sr, mono=True)
	audio = audio.reshape(-1, 1).T[0].T
	return audio, mu_law_encode(audio)

def create_audio(filenames, sample_rate=8000):
	data = []
	audios = []
	for filename in filenames:
		print('loading ', filename)
		audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
		#rate, data = wavfile.read(filename)
		#print(rate);print(data);exit()
		audio = audio.reshape(-1, 1).T[0].T
		audios.append(audio[:, None])
	#print(data.shape);print(audio[:, None].shape);exit()
	return audios

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

def cross_entropy(prob_of_gt_sample, q_channels=256):
	return -np.log(prob_of_gt_sample)

def entropy(pred):
	return stats.entropy(pred)

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
	DEVICE_ID_LIST = GPUtil.getFirstAvailable(order = 'last', maxLoad=0.85, verbose=True)
	DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

	os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
	log('Preparing environment by choosing a gpu {} and setting resource limit={}'.format(DEVICE_ID, resource_limit))
	soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
	resource.setrlimit(resource.RLIMIT_CPU, (resource_limit, hard))

def save_weights(self, saver, outdir, epoch, iteration, sess, loss, log):
	create_out_dir(outdir, log)
	checkpoint_dir = outdir + 'saved_weights/'
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_path = checkpoint_dir + timestamp() + '_epoch' + epoch + '_loss=' + loss + '_model.ckpt'
	log('Storing checkpoint as {} ...'.format(checkpoint_path))
	saver.save(sess, checkpoint_path, global_step=iteration, write_meta_graph=False)

def plot_gaussian_distr(outdir, name, prediction, chosen_sample, gt, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(10, 4))
		plt.plot(np.arange(256), prediction)
		#print(chosen_sample);print(prediction[chosen_sample])
		plt.scatter(chosen_sample, prediction[chosen_sample], color=['green'])
		if gt:
			plt.scatter(gt, prediction[gt], color=['red'])
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		#plt.xlim(np.arange(-1, 1, 256))
		plt.xlabel('bin')
		plt.ylabel('probability')
		plt.savefig(outdir + name, dpi=100)

def plot_waveform(outdir, name, data, sr, should_plot, log):
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


'''
def plot_waveform(outdir, name, data, data2, div, sr, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		times = np.arange(len(data)+len(data2))/float(sr)
		fig = plt.figure(figsize=(60, 4))
		ax = fig.add_subplot(111)
		ax.plot(times[:len(data)], data, color='blue')
		ax.plot(times[len(data):], data2, color='orange')
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name, dpi=100)
'''
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
		fig = plt.figure(figsize=(80, 8))
		ax1 = fig.add_subplot(111)
		ax1.plot(times,gt, label='ground truth') 
		#ax1.plot(times,random_data, label='predictions (random)') 
		ax1.plot(times,data, label='predictions (argmax)') 
		plt.legend(loc='upper right')
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		log('Saving plot of the waveform as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name, dpi=100)

def plot_spectogram(outdir, name, data, sr, should_plot, log):
	if should_plot:
		#create_out_dir(outdir, log)
		plt.figure(figsize=(30, 10))
		D = librosa.amplitude_to_db(librosa.core.magphase(librosa.stft(data))[0])
		#CQT = librosa.amplitude_to_db(librosa.cqt(data, sr=sr), ref=np.max)
		'''
		+		plt.subplot(4, 2, 3)
+		librosa.display.specshow(CQT, y_axis='cqt_note')
+		plt.colorbar(format='%+2.0f dB')
+		plt.title('Constant-Q power spectrogram (note)')
		'''
		#plt.subplot(4, 2, 1)
		librosa.display.specshow(D, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Linear-frequency power spectrogram')
		#plt.title('Linear power spectrogram (grayscale)')
		log('Saving spectogram as \'{}\''.format(outdir + name))
		plt.savefig(outdir + name)

def plot_entropy(outdir, name, entropies, spacing_int, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(24, 12))
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
		plt.legend(loc='upper right')
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