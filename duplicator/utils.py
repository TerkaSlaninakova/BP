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

plt.switch_backend('agg')
CURRENT_RUN_TIMESTAMP = None

'''
For experimentation with preprocessing:

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out

def mu_law_decode(output, channels=256):
    
    signal = 2 * (np.float32(output) / channels) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / channels) * ((1 + channels)**abs(signal) - 1)
    return np.sign(signal) * magnitude

def mu_law_encode(audio, channels):
    template = np.linspace(-1, 1, channels)
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(channels * safe_audio_abs) / np.log1p(channels)
    signal = np.sign(audio) * magnitude
    return signal
'''

def create_logspace_template(n_channels=256):
    pos = list(np.logspace(-10, 0.00000001, 128, base=math.e).reshape(-1, 1).T[0])
    neg = list(reversed(-np.logspace(-10, 0.00000001, 128, base=math.e).reshape(-1, 1).T[0]))
    template = np.array(neg + pos)
    return template
'''
def read_audio(path, sample_rate, outdir):
	audio_, _ = librosa.load(path, sr=sample_rate, mono=True)
	duration = librosa.core.get_duration(audio_, sr=sample_rate)
	return audio_.reshape(-1, 1).T[0].T, duration
'''
def create_batch_dir(data_dir, sample_rate, channels, log):
    l_inputs = []
    l_targets = []
    audio_files = list(glob.iglob(data_dir + '*.wav'))
    audio = None
    duration = 0
    log('Reading audio from {}, found {}'.format(data_dir, len(audio_files)))
    for filename in audio_files:
        inputs, targets, audio, duration = create_batch(filename, sample_rate, channels, log)
        l_inputs.append(inputs)
        l_targets.append(targets)
    return l_inputs, l_targets, audio, duration

def create_batch(path, sample_rate, channels, log):
	audio_, _ = librosa.load(path, sr=sample_rate, mono=True)
	duration = librosa.core.get_duration(audio_, sr=sample_rate)
	audio = audio_.reshape(-1, 1).T[0].T
	# Quantization of the audio to 'channels' different amplitudes, representing the sample rate of the final output
	log('Quantized the audio to {} different amplitudes'.format(channels))
	template = create_logspace_template(channels)
	bins = np.digitize(audio[0:-1], template) - 1
	inputs_ = template[bins]
	inputs = inputs_[None, :, None]
	targets = np.digitize(audio[1::], template) - 1
	outdir = os.path.dirname(os.path.realpath(__file__)) + '/'
	return inputs, targets[None, :], audio, duration

def write_data(outdir, name, data, output_sample_rate, log):
	create_out_dir(outdir, log)
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

def plot_losses(outdir, name, losses, iterations, should_plot, log):
	if should_plot:
		create_out_dir(outdir, log)
		plt.figure(figsize=(12, 6))
		iterations_range = [i for i in range(iterations)]
		plt.plot(iterations_range, losses)

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