# Playing with WaveNet implementations found around the internet

### What was done during the experimentation:
Trained and gathered results from WaveNet implementation in Keras, in Tensorflow, the 'fast' implementation in TF and in text-oriented wavenet in TF. The experimentation with sound was done on VCTK data as well as music (IRMAS). This repo contains only the final results after specific number of steps.
Training was done on VCTK corpus, music (piano on [IRMAS](https://www.upf.edu/web/mtg/irmas)) and text files, generated wavs corresponding to different number of steps in the training phase

### What was not done:
There is no direct implementation having the text-to-speech functionality integrated, see [this link](https://github.com/ibab/tensorflow-wavenet/issues/252) - that wasn't tried. Neither was adjusting the wavenet architecture and comparing results (no time for that since)
Did not have time for [Speech to text](https://github.com/buriburisuri/speech-to-text-wavenet) neither.

## [Wavenet in Keras](https://github.com/basveeling/wavenet) 
Very computationally expensive, Tesla K80 is just not enough to train this, generation of 1s of audio takes 4 minutes according to their README, but training even to 100 epochs is almost impossible, trained to 30 epochs with unconvincing results. I used the default data provided in data/ folder.

## [Wavenet in Tensorflow](https://github.com/ibab/tensorflow-wavenet)
Primary tensorflow implementation of wavenet. I used the [VCTK data](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html), only from 1 person (p340) to minimize the training time (423 wavs in training) the loss becomes stuck around 1.5 from like 200th iteration up to 40 000, which was the maximum i went to, yet the model is progressing (hear the wavs) and with the number of iterations. The training was very long, but with the checkpoints i was able to incrementally train it to produce something, which is promising, yet still far from the deepmind's results.
I also tried to train on music wavs to see what comes out, after approximatelly 15000 iterations, the generated audio contained mostly noise

## [Fast-Wavenet in Tensorflow](https://github.com/tomlepaine/fast-wavenet)
Implementation focusing on speeding up the generation (not training). The model was run for little over 4 hours, was trained on a single wav file and was able to directly reconstruct the same waveform. Run through jupyter notebook to hear the generated output. Fast-wavenet is now completely integrated into https://github.com/ibab/tensorflow-wavenet according to their README, that could explain significantly faster training time compared to keras' model.

## [Tex-Wavenet in Tensorflow](https://github.com/Zeta36/tensorflow-tex-wavenet)
Uses wavenet model to generate text, reusing tensorflow-wavenet. When trained on the default data with 108000 iterations (it was quite fast) on complete works of Shakespeare, with loss of about 1.3, it created interesting results
