import tensorflow as tf
import librosa
import argparse
import os 
import numpy as np
from utils import  *
from train import Trainer
from generate import Generator

TRAIN_EPOCHS = 300
TARGET_LOSS = 0.001
DATA_DIR = '/pub/tmp/xslani06/wavenet/data/vctk/wav48/p225/'
OUTPUT_DIR = ''
LEARNING_RATE = 1e-4
GPU_FRACTION = 0.8
SAMPLE_RATE = 48000
N_DILATIONS = 10
N_BLOCKS = 5
Q_CHANNELS = 256
PLOT = True
LOG = True
SHOULD_TRAIN = True
DIL_WIDTH = 64
RES_WIDTH = 64
SKIP_WIDTH = 1024
KERNEL_WIDTH = 2
RESOURCE_LIMIT = 27000
OUT_SAMPLES = 16000
SEED_FROM = '/pub/tmp/xslani06/wavenet/data/vctk/wav48/p225/p225_001.wav' #"/pub/tmp/xslani06/wavenet/data/magna_piano_out/360000.wav" #'/pub/tmp/xslani06/wavenet/data/vctk/wav48/p225/p225_001.wav' #"/pub/tmp/xslani06/wavenet/data/magna_piano_out/360000.wav"
#SEED_FROM = "/pub/tmp/xslani06/wavenet/data/yt_wav_out/9_768000.wav"
# allows for boolean arg parsing
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet')
    parser.add_argument('--train_epochs', type=int, default=TRAIN_EPOCHS, help='Max number of epochs to be used for training, default: ' + str(TRAIN_EPOCHS))
    parser.add_argument('--target_loss', type=float, default=TARGET_LOSS, help='Which loss should the training stop on: ' + str(TARGET_LOSS))
    parser.add_argument('--q_channels', type=int, default=Q_CHANNELS, help='How many distinct amplitudes (quantization channels) should be recognized during the training process, default: ' + str(Q_CHANNELS))
    parser.add_argument('--dil_w', type=int, default=DIL_WIDTH, help='How many kernels to learn for the dilated convolution, default: ' + str(DIL_WIDTH))
    parser.add_argument('--res_w', type=int, default=RES_WIDTH, help='How many kernels to learn for the residual block, default: ' + str(RES_WIDTH))
    parser.add_argument('--kernel_w', type=int, default=KERNEL_WIDTH, help='With of the kernel window, default: ' + str(KERNEL_WIDTH))
    parser.add_argument('--skip_w', type=int, default=SKIP_WIDTH, help='How many kernels to learn for the skip output, default: ' + str(SKIP_WIDTH))
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE, help='Sample rate with which the input wav should be read, default ' + str(SAMPLE_RATE))
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to a directory with training data')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate to be used in training, default: ' + str(LEARNING_RATE))
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory for outputs (wavs, plots, ...), default: ' + str(OUTPUT_DIR))
    parser.add_argument('--gpu_fraction', type=float, default=GPU_FRACTION, help='Percentage of GPU to be used, default: ' + str(GPU_FRACTION))
    parser.add_argument('--plot', type=str2bool, default=PLOT, help='Bool to decide whether to plot output data' + str(PLOT))
    parser.add_argument('--n_dilations', type=int, default=N_DILATIONS, help='How many successive dilations should the net consist of, default: ' + str(N_DILATIONS))
    parser.add_argument('--n_blocks', type=int, default=N_BLOCKS, help='How many blocks of dilations should be there, default: ' + str(N_BLOCKS))
    parser.add_argument('--out_samples', type=int, default=OUT_SAMPLES, help='Number of output samples, default: ' + str(OUT_SAMPLES))
    parser.add_argument('--seed_from', type=str, default=SEED_FROM, help='Wav file to seed the generation from' + str(SEED_FROM))
    parser.add_argument('--log', type=bool, default=LOG, help='Should log: ' + str(LOG))
    parser.add_argument('--model', type=str, default=None, help='Directory of the saved model to restore for further training, default: None')
    parser.add_argument('--train', type=str2bool, default=SHOULD_TRAIN, help='Decides whether to start training the model (form scratch or saved) or proceed to generation. Default: ' + str(SHOULD_TRAIN))
    parser.add_argument('--c', type=str, default="", help='Commentary on the run - gets appended to the output directory name')
    return parser.parse_args()

if __name__ == '__main__':
    parser = get_arguments()
    parser.output_dir = os.path.dirname(os.path.realpath(__file__)) + '/' + timestamp() + parser.c + '/' if parser.output_dir == '' else parser.output_dir
    log = Log(should_log=parser.log).log
    log('Got arguments: {}'.format(parser))
    prepare_environment(RESOURCE_LIMIT, log)

    losses = []
    trainer = Trainer(
        q_channels=parser.q_channels,
        dil_width=parser.dil_w,
        res_width=parser.res_w,
        skip_width=parser.skip_w,
        kernel_width=parser.kernel_w,
        n_dilations=parser.n_dilations, 
        n_blocks=parser.n_blocks, 
        learning_rate=parser.learning_rate,
        log=log)
    if parser.train:
        train_files, validation_files = prepare_datasets(parser.data_dir, log)
        train_batch = create_audio(train_files, parser.sample_rate)
        val_batch = create_audio(validation_files, parser.sample_rate)
        losses = trainer.train(
            train_batch=train_batch,
            val_batch=val_batch,
            target_loss=parser.target_loss, 
            train_epochs=parser.train_epochs, 
            output_dir=parser.output_dir,
            should_plot=parser.plot, 
            load_dir=parser.model,
            log=log)
    else:
        log('Skipping training')
    
    if losses != [] or not parser.train:
        generator = Generator(trainer)
        final_predictions = generator.generate(parser.model, parser.out_samples, parser.seed_from, parser.output_dir, log)
        final_predictions = np.array(final_predictions)
        plot_waveform(parser.output_dir, 'out_'+timestamp()+'.png', final_predictions, 1000, parser.plot, log)
        plot_spectogram(parser.output_dir, 'out_spectograms_'+timestamp()+'.png', final_predictions, 1000, parser.plot, log)
        write_data(parser.output_dir, 'pred_'+timestamp()+'.wav', final_predictions, parser.sample_rate, log)
    else:
        log('Skipping generation')
    