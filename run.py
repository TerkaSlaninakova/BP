'''
File: run.py
Author: Terézia Slanináková (xslani06)

Entry-point to the codebase of a custom WaveNet implementation.
Uses params.json for parameter specification and initializes the training and generation process.

Usage:
    python3 run.py [--train=true|false] [--generate=true|false] [--params=parameters.json]
'''
import tensorflow as tf
import librosa
import argparse
import os 
import numpy as np
import json
from utils import  *
from train import Trainer
from generate import Generator
import shutil

PARAMETERS_PATH = './parameters.json'
TRAIN = True
GENERATE = True

def parse_parameters(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='WaveNet')
    parser.add_argument('--params', type=str, default=PARAMETERS_PATH, help='Path to json parameters, default: ' + str(PARAMETERS_PATH))
    parser.add_argument('--train', type=str2bool, default=TRAIN, help='If training should be run' + str(TRAIN))
    parser.add_argument('--generate', type=str2bool, default=GENERATE, help='If generating should be run' + str(GENERATE))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    params = parse_parameters(args.params)
    params['output_dir'] = create_output_dir(params['output_dir'])
    log = Log(should_log=params['log']).log
    create_out_dir(params['output_dir'] ,log)
    shutil.copyfile(PARAMETERS_PATH, params['output_dir'] + 'parameters.json')
    
    log('Proceeding with arguments: {}'.format(params))
    prepare_environment(params['resource_limit'], log)
    
    trainer = Trainer(
        q_channels=params['wavenet']['bit_depth'],
        dil_width=params['wavenet']['dilation_ch'],
        res_width=params['wavenet']['residual_ch'],
        skip_width=params['wavenet']['skip_ch'],
        kernel_width=params['wavenet']['kernel_width'],
        n_dilations=params['wavenet']['n_dilations'],
        n_blocks=params['wavenet']['n_blocks'],
        learning_rate=params['train']['learning_rate'],
        log=log)
    
    losses = []
    if args.train:
        train_files, validation_files = prepare_datasets(params['dataset_dir'], log)
        train_batch = create_audio(train_files, params['sampling_rate'])
        val_batch = create_audio(validation_files, params['sampling_rate'])
        losses = trainer.train(
            train_batch=train_batch,
            val_batch=val_batch,
            train_epochs=params['train']['epochs'], 
            output_dir=params['output_dir'],
            should_plot=params['plot'], 
            load_dir=params['model'],
            log=log)
    else:
        log('Skipping training')
    
    if losses != [] or not args.train:
        generator = Generator(trainer, params['output_dir'], log)
        generated_waveform = generator.generate(
            params['model'], params['generate']['out_samples'],params['generate']['seed_from'], params['generate']['teacher_forcing'], params['sampling_rate'], params['output_dir'], params['log'])
        generated_waveform_ = np.array(generated_waveform)
        plot_waveform(params['output_dir'], 'out_'+timestamp()+'.png', generated_waveform, len(generated_waveform), params['plot'], log)
        plot_spectogram(params['output_dir'], 'out_spectograms_'+timestamp()+'.png', generated_waveform, len(generated_waveform), params['plot'], log)
        write_data(params['output_dir'], 'pred_'+timestamp()+'.wav', generated_waveform, params['sampling_rate'], log)
    else:
        log('Skipping generating')
    