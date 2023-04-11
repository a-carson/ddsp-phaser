## Differentiable Grey-box Modelling of Phaser Effects using Frame-based Spectral Processing

Accompanying code for the DAFx23 submission by A. Carson, C. Valentini-Botinhao, S. King and S. Bilbao.

The audio dataset used in the paper can be found under ``audio_data/``.

Audio examples can be found [here](https://a-carson.github.io/ddsp-phaser/).

### Example Usage
#### Getting started
First, clone the repo into the desired directory:
```
git clone git@github.com:a-carson/ddsp-phaser.git 
```
You will then need to install the relevant Python libraries.

#### Training
To train with default arguments use:
````
python3 train_phaser.py 
````
For full list of arguments use ``python3 train_phaser.py --help`` or see ``train_phaser.py`` file.

#### Inference
An example pre-trained model of the EHX Small Stone under parameter configuration SS-A is provided.
To run this model in inference mode use:
```
python3 inference.py --checkpoint_path checkpoints/ss-A.ckpt
```
This will save an audio file ``inference_out.wav`` in the working directory.

The LFO rate (in Hz) can be changed at inference using the 
``--f0`` flag, for example:
```
python3 inference.py --checkpoint_path checkpoints/ss-A.ckpt --f0 0.5
```
