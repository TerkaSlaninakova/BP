import datetime
import time
import librosa
from matplotlib import pyplot as plt
import numpy as np

plt.switch_backend('agg')

LOG = False

def _create_template(data, n_classes):
	'''
	Creates template of amplitudes according to n_classes with dimensions of read audio: (data.shape, )
	'''
	template = np.linspace(-1, 1, n_classes)
	inputs = np.digitize(data, template)
	return template, inputs

def create_targets(data, n_classes=256):
	'''
	Creates targets for the net of shape (1, data.shape)
	'''
	_, targets = _create_template(data, n_classes)
	return targets[None, :]

def create_inputs(data, n_classes=256):
	'''
	Creates the final representation of input data of shape (1, samples in read audio, 1) for the net
	'''
	template, inputs = _create_template(data, n_classes)
	return template[inputs][None,:,None]

def read_audio(path, sample_rate):
	audio, _ = librosa.load(path, sr=sample_rate, mono=True)
	audio = audio.reshape(-1, 1)
	# getting rid of an extra dimension that was added by reshape()
	return audio.T[0].T

def write_data(path, data, output_sample_rate):
	librosa.output.write_wav(path, data, output_sample_rate)

def timestamp():
	return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

class Log:
	def __init__(self,should_log):
		self.should_log = should_log
	
	def log(self, name, object_to_log=None):
		if self.should_log:
			if object_to_log is not None and LOG:
				print("[D] {}: {} {}".format(name,object_to_log.dtype,object_to_log.get_shape()))
			elif LOG:
				print("[D] {}".format(name))

def plot(name, data, sr, should_plot):
	if should_plot:
		times = np.arange(len(data))/float(sr)
		plt.figure(figsize=(30, 4))
		plt.fill_between(times,data) 
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		plt.savefig(name, dpi=100)
