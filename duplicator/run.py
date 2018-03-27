import tensorflow as tf
import librosa
import argparse
import os 

from utils import  *
from train import Net
from generate import Generator

TRAIN_ITERATIONS = 10000
TARGET_LOSS = 0.0001
PATH_TO_WAV = 'data/yes.wav'
DATA_DIR = '/pub/tmp/xslani06/wavenet/data/irmas/test/'
OUTPUT_DIR = ''
LEARNING_RATE = 0.0005
GPU_FRACTION = 0.5
SAMPLE_RATE = 8000
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
    parser.add_argument('--target_loss', type=float, default=TARGET_LOSS, help='Which loss should the training stop on: ' + str(TARGET_LOSS))
    parser.add_argument('--n_classes', type=int, default=N_CLASSES, help='How many distinct amplitudes should be recognized during the training process, default: ' + str(N_CLASSES))
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE, help='Sample rate with which the input wav should be read, default ' + str(SAMPLE_RATE))
    parser.add_argument('--path_to_wav', type=str, default=PATH_TO_WAV, help='Path to a single wav file (current limitation)')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate to be used in training, default: ' + str(LEARNING_RATE))
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory for outputs (wavs, plots, ...), default: ' + str(OUTPUT_DIR))
    parser.add_argument('--gpu_fraction', type=float, default=GPU_FRACTION, help='Percentage of GPU to be used, default: ' + str(GPU_FRACTION))
    parser.add_argument('--plot', type=str2bool, default=PLOT, help='Bool to decide whether to plot output data' + str(PLOT))
    parser.add_argument('--n_dilations', type=int, default=N_DILATIONS, help='How many successive dilations should the net consist of, default: ' + str(N_DILATIONS))
    parser.add_argument('--n_blocks', type=int, default=N_BLOCKS, help='How many blocks of dilations should be there, default: ' + str(N_BLOCKS))
    parser.add_argument('--n_hidden', type=int, default=N_HIDDEN, help='How many hidden layers: ' + str(N_BLOCKS))
    parser.add_argument('--log', type=bool, default=LOG, help='Should log: ' + str(LOG))
    parser.add_argument('--model', type=str, default=None, help='Directory of the saved model to restore for further training, default: None')
    parser.add_argument('--train', type=str2bool, default=SHOULD_TRAIN, help='Decides whether to start training the model (form scratch or saved) or proceed to generation. Default: ' + str(SHOULD_TRAIN))
    parser.add_argument('--save_weights', type=str2bool, default=SHOULD_SAVE_WEIGHTS, help='Decides whether to save weights. Default: ' + str(SHOULD_SAVE_WEIGHTS))
    parser.add_argument('--c', type=str, default="", help='Commentary on the run - gets appended to the output directory name')
    return parser.parse_args()

if __name__ == '__main__':
    parser = get_arguments()
    parser.output_dir = os.path.dirname(os.path.realpath(__file__)) + '/' + timestamp() + parser.c + '/' if parser.output_dir == '' else parser.output_dir
    
    log = Log(should_log=parser.log).log
    log('Got arguments: {}'.format(parser))

    if parser.train and parser.model is not None:
        parser.train = False
        log('Assigning False to --train since --model was provided')

    log('Read audio from \'{}\''.format(parser.path_to_wav))
    inputs, targets, audio, duration = create_batch_dir(DATA_DIR, parser.sample_rate, parser.n_classes, log)
    log('Created inputs, number of batches: \'{}\' shape: \'{}\' and targets, shape: \'{}\''.format(len(inputs), inputs[0].shape, targets[0].shape))
    model = Net(
        n_samples=inputs[0].shape[1],
        n_classes=parser.n_classes, 
        n_dilations=parser.n_dilations, 
        n_blocks=parser.n_blocks, 
        n_hidden=parser.n_hidden, 
        gpu_fraction=parser.gpu_fraction,
        learning_rate=parser.learning_rate,
        log=log)
    if parser.train:
        costs = model.train(
            inputs=inputs, 
            targets=targets, 
            target_loss=parser.target_loss, 
            train_iterations=parser.train_iters, 
            output_dir=parser.output_dir,
            should_plot=parser.plot, 
            should_save_weights=parser.save_weights,
            load_dir=parser.model,
            n_of_batches=len(inputs),
            log=log)
    else:
        log('Skipping training')

    gen = Generator(model)
    output_length = int(parser.sample_rate * duration)
    final_predictions = gen.run(inputs[0][:, 0:1, 0], output_length, inputs[0][0, :, 0], parser.model)
    create_out_dir(parser.output_dir, log)
    plot_waveform(parser.output_dir, 'out_'+timestamp()+'.png', final_predictions, output_length, parser.plot, log)
    plot_spectogram(parser.output_dir, 'out_spectograms_'+timestamp()+'.png', final_predictions, output_length, parser.plot, log)
    write_data(parser.output_dir, 'pred_'+timestamp()+'.wav', final_predictions, parser.sample_rate, log)