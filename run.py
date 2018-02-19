import tensorflow as tf
import librosa
import argparse
import os 

from utils import  *
from train import Net
from generate import Generator

TRAIN_ITERATIONS = 7000
TARGET_COST = 0.1
PATH_TO_WAV = 'data/p225_001.wav'
OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
LEANING_RATE = 0.001
GPU_FRACTION = 0.5
SAMPLE_RATE = 32000
N_DILATIONS = 14
N_BLOCKS = 2
N_HIDDEN = 128
N_CLASSES = 256
PLOT = False
LOG = False
SHOULD_TRAIN = True
SHOULD_SAVE_WEIGHTS = False

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
    parser.add_argument('--train_iters', type=int, default=TRAIN_ITERATIONS, help='Max number of iterations to be used for training, default: ' + str(TRAIN_ITERATIONS))
    parser.add_argument('--target_cost', type=float, default=TARGET_COST, help='Which cost should the training stop on: ' + str(TARGET_COST))
    parser.add_argument('--n_classes', type=int, default=N_CLASSES, help='How many distinct amplitudes should be recognized during the training process, default: ' + str(N_CLASSES))
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE, help='Sample rate with which the input wav should be read, default ' + str(SAMPLE_RATE))
    parser.add_argument('--path_to_wav', type=str, default=PATH_TO_WAV, help='Path to a single wav file (current limitation)')
    #parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate to be used in training, default: ' + str(LEARNING_RATE))
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory for outputs (wavs, plots, ...), default: ' + str(OUTPUT_DIR))
    parser.add_argument('--gpu_fraction', type=float, default=GPU_FRACTION, help='Percentage of GPU to be used, default: ' + str(GPU_FRACTION))
    parser.add_argument('--should_plot', type=str2bool, default=PLOT, help='Bool to decide whether to plot output data' + str(PLOT))
    parser.add_argument('--n_dilations', type=int, default=N_DILATIONS, help='How many successive dilations should the net consist of, default: ' + str(N_DILATIONS))
    parser.add_argument('--n_blocks', type=int, default=N_BLOCKS, help='How many blocks of dilations should be there, default: ' + str(N_BLOCKS))
    parser.add_argument('--n_hidden', type=int, default=N_HIDDEN, help='How many hidden layers: ' + str(N_BLOCKS))
    parser.add_argument('--log', type=bool, default=LOG, help='Should log: ' + str(LOG))
    parser.add_argument('--saved_model_dir', type=str, default=None, help='Directory of the saved model to restore for further training, default: None')
    parser.add_argument('--should_train', type=str2bool, default=SHOULD_TRAIN, help='Decides whether to start training the model (form scratch or saved) or proceed to generation. Default: ' + str(SHOULD_TRAIN))
    parser.add_argument('--should_save_weights', type=str2bool, default=SHOULD_SAVE_WEIGHTS, help='Decides whether to save weights. Default: ' + str(SHOULD_SAVE_WEIGHTS))
    return parser.parse_args()

if __name__ == '__main__':
    parser = get_arguments()
    log = Log(should_log=parser.log).log
    log('[D] Got arguments: {}'.format(parser))

    if not parser.should_train and parser.saved_model_dir is None:
        print('Either provide directory to load saved model from or allow training, exiting.');exit()

    data = read_audio(parser.path_to_wav, parser.sample_rate)
    log('[D] Read {} of shape {}'.format(parser.path_to_wav, data.shape));
    plot_waveform(parser.output_dir + 'in_'+timestamp()+'.png', data, parser.sample_rate, parser.should_plot)
    plot_spectogram(parser.output_dir + 'in_spectograms_'+timestamp()+'.png', data, parser.sample_rate, parser.should_plot)

    inputs = create_inputs(data, parser.n_classes)
    targets = create_targets(data, parser.n_classes)

    model = Net(
        n_samples=inputs.shape[1], 
        n_classes=parser.n_classes, 
        n_dilations=parser.n_dilations, 
        n_blocks=parser.n_blocks, 
        n_hidden=parser.n_hidden, 
        gpu_fraction=parser.gpu_fraction)
    
    if parser.should_train:
        model.train(inputs, targets, parser.target_cost, parser.train_iters, parser.output_dir, parser.should_plot, parser.should_save_weights, parser.saved_model_dir)

    gen = Generator(model)
    inputs_ = inputs[:, 0:1, 0]
    final_predictions = gen.run(inputs_, data.shape[0], parser.saved_model_dir)

    plot_waveform(parser.output_dir + 'out_'+timestamp()+'.png', final_predictions, parser.sample_rate, parser.should_plot)
    plot_spectogram(parser.output_dir + 'out_spectograms_'+timestamp()+'.png', final_predictions, parser.sample_rate, parser.should_plot)
    write_data(parser.output_dir + 'pred_'+timestamp()+'.wav', final_predictions, parser.sample_rate)
