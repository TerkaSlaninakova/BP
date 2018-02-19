import datetime
import time
import librosa
from matplotlib import pyplot as plt
import numpy as np
import librosa.display

plt.switch_backend('agg')
CURRENT_RUN_TIMESTAMP = None
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
			elif LOG:
				print("[D] {}".format(name))

def plot_waveform(name, data, sr, should_plot):
	if should_plot:
		times = np.arange(len(data))/float(sr)
		plt.figure(figsize=(30, 4))
		plt.fill_between(times,data) 
		plt.xlim(times[0], times[-1])
		plt.xlabel('time (s)')
		plt.ylabel('amplitude')
		plt.savefig(name, dpi=100)

def plot_spectogram(name, data, sr, should_plot):
	if should_plot:
		plt.figure(figsize=(12, 8))
		D = librosa.amplitude_to_db(librosa.stft(data), ref=np.max)

		plt.subplot(4, 2, 1)
		librosa.display.specshow(D, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Linear-frequency power spectrogram')

		plt.subplot(4, 2, 2)
		librosa.display.specshow(D, y_axis='log')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Log-frequency power spectrogram')

		CQT = librosa.amplitude_to_db(librosa.cqt(data, sr=sr), ref=np.max)

		plt.subplot(4, 2, 3)
		librosa.display.specshow(CQT, y_axis='cqt_note')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Constant-Q power spectrogram (note)')

		plt.subplot(4, 2, 4)
		librosa.display.specshow(CQT, y_axis='cqt_hz')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Constant-Q power spectrogram (Hz)')

		C = librosa.feature.chroma_cqt(y=data, sr=sr)
		plt.subplot(4, 2, 5)
		librosa.display.specshow(C, y_axis='chroma')
		plt.colorbar()
		plt.title('Chromagram')

		plt.subplot(4, 2, 6)
		librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Linear power spectrogram (grayscale)')

		plt.savefig(name)

def plot_costs(name, costs, iterations, should_plot):
	print(costs)
	if should_plot:
		plt.figure(figsize=(12, 6))
		iterations_range = [i for i in range(iterations)]
		plt.plot(iterations_range, costs)

		plt.xlabel('iterations')
		plt.ylabel('costs')
		plt.title('Training process')
		plt.grid(True)
		plt.savefig(name)
