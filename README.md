# Wavenet
- Training phase: Creates the model as a stack of dilated causal convolution layers
- Generation phase: Inspired by [fast-wavenet](https://arxiv.org/pdf/1611.09482.pdf) - caching is used to avoid recomputing ceratin outputs for every generated sample 

## Structure
- run.py -> entry point of the program, handles arguments, invokes all the actions
- train.py -> class Net representing the WaveNet model
- generate.py -> class Generate representing the Generator
- utils.py -> all the other useful utilities, such as plotting, logging and timestamp creation

## Examples
I trained the net on 3 simple (1-2s long) wavs, bed.wav and yes.wav are from the [speech recognition challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data) p225_001.wav is form the [vctk corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

## Output
- generated wav saved in --output_dir named pred_[timestamp].wav
- if --plot is set to True (False by default), a waveform plot of input wav and generated wav

## TODOs
- Support for training with multiple wavs
- Support for longer wavs
- Support for weight-saving after training