# Wavenet
- Preprocessing phase: Audio is read by librosa with a chosen sample rate and reshaped to [-1, 1]. It is then quantized to --n_channels (256 by default) quantization channels to create the inputs and targets used as training data.
- Training phase: Model is created as a stack of dilated causal convolution layers, AdamOptimizer is created to be used for training. Training stops after reaching completing desired number of epochs. It's also possible to stop training on Ctrl+C signal. Training can be also skipped altogether assuming --model= argument is supplied with directory containing saved weights of a model that should be restored for generation.
- Generation phase: Repurposed from [fast-wavenet](https://arxiv.org/pdf/1611.09482.pdf) - caching is used to avoid recomputing outputs for every generated sample. After being supplied with the first sample the generation model predict every next sample from the previous one using distributions of conditional probabilities.

## Structure
- run.py -> entry point of the program, handles arguments, invokes all the actions
- train.py -> class Net representing the WaveNet model
- generate.py -> class Generate representing the Generator
- utils.py -> all the other useful utilities, such as plotting, logging and timestamp creation

## Data
I trained the net on 3 simple (1-2s long) wavs, `bed.wav` and `yes.wav` are from the [speech recognition challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data) `p225_001.wav` is form the [vctk corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

## Output
- generated wav saved in `--output_dir` named `pred_[timestamp].wav`
- if `--plot` is set to True (False by default), a waveform plot of input wav and generated wav

## How to use
Have running python3 and environment, where additional packages can be installed. Miniconda/Anaconda/virtualenv-type of environments can be helpful. 
Install all the necessary packages by running `pip install -r requirements.txt`
Run `python3 run.py --model=trained/2018-03-13_18-39-37/saved_weights/ --train=False` for generation of wav from pre-trained model (no GPU required). For training and generating, default `python3 run.py` suffices

