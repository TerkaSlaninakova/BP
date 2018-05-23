# WaveNet for music generation
[WaveNet](https://arxiv.org/abs/1609.03499) is a generative model for raw audio waveforms, introduced by van den Oord et al in 2016, that has since found many applications in multiple domains. The purpose of this repo is to form a codebase that would use wavenet to generate music. 

## How to use
Prerequisites:
  - python3 and possibility of installing additional packages (Miniconda/Anaconda/virtualenv can be helpful)
Install all the necessary packages by running `pip install -r requirements.txt`

Training and generating
  - Configure the run with `parameters.json` and run `python3 run.py`

Generating only:
  - I included one pre-trained model for generating only, no GPU required: 
Run `python3 run.py --model=trained/2018-03-13_18-39-37/saved_weights/ --train=False` for generation of wav from pre-trained model (no GPU required). For training and generating, default `python3 run.py` suffices

## Structure
- run.py -> entry point of the program, handles arguments, invokes all the actions
- train.py -> class Train involving training operations
- wavenet.py -> class Net representing the WaveNet model
- generate.py -> class Generate representing the Generator
- utils.py -> all the other useful utilities, such as plotting, logging and timestamp creation

## Process
- Preprocessing phase: Audio is read by librosa with a chosen sample rate and reshaped to [-1, 1]. It is then quantized to --n_channels (256 by default) quantization channels to create the inputs and targets used as training data.
- Training phase: Repurposed from [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet). Model is created as a stack of dilated causal convolution layers, AdamOptimizer is created to be used for training. Training stops after reaching completing desired number of epochs. Training can be skipped altogether assuming --model= argument is supplied with directory containing saved weights of a model that should be restored for generation.
- Generation phase: Repurposed from [fast-wavenet](https://arxiv.org/pdf/1611.09482.pdf) - caching is used to avoid recomputing outputs for every generated sample. After being supplied with the first sample the generation model predict every next sample from the previous one using distributions of conditional probabilities.

## Data
These datasets were tested:
[vctk corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
[YT-8M](https://research.google.com/youtube8m/)
[MagnaTagATune](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
[Irmas](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)

## Output
- generated wav saved in `--output_dir` named `pred_[timestamp].wav`
- if `--plot` is set to True (False by default), a waveform plot of input wav and generated wav
