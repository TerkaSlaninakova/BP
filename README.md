# Wavenet
- Preprocessing phase: Audio is read by librosa with a chosen sampling rate, then quantizedusing mu-law.
- Training phase: Uses [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet), model is created as a stack of dilated causal convolution layers, AdamOptimizer is created to be used for training with supplied --learning_rate. Training stops after reaching desired --target_loss or --train_iteration. If --save_weights is set to True (False by default), weights are continuously saved during training. It's also possible to stop training on Ctrl+C signal. Training can be also skipped altogether assuming --model= argument is supplied with directory containing saved weights of a model that should be restored for generation.
- Generation phase: Uses by [fast-wavenet](https://arxiv.org/pdf/1611.09482.pdf) - caching is used to avoid recomputing cerating outputs for every generated sample. After being supplied with the first sample the generation model predict every next sample from the previous one using distributions of conditional probabilities.

## Examples of usage
Run `python3 run.py --model=trained/2018-03-13_18-39-37/saved_weights/ --train=False` for generation of wav from pre-trained model (no GPU required). For training and generating, default `python3 run.py` suffices

