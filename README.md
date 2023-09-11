## Differentiable Grey-box Modelling of Phaser Effects using Frame-based Spectral Processing

Accompanying code for the DAFx23 submission by A. Carson, C. Valentini-Botinhao, S. King and S. Bilbao.

The audio dataset used in the paper can be found under ``audio_data/``.

Audio examples can be found [here](https://a-carson.github.io/ddsp-phaser/).

### Abstract
Machine learning approaches to modelling analog audio effects have seen intensive investigation in recent years, particularly in the context of non-linear time-invariant effects such as guitar amplifiers. For modulation effects such as phasers, however, new challenges emerge due to the presence of the low-frequency oscillator which controls the slowly time-varying nature of the effect. Existing approaches have either required foreknowledge of this control signal, or have been non-causal in implementation. This work presents a differentiable digital signal processing approach to modelling phaser effects in which the underlying control signal and time-varying spectral response of the effect are jointly learned. The proposed model processes audio in short frames to implement a time-varying filter in the frequency domain, with a transfer function based on typical analog phaser circuit topology. We show that the model can be trained to emulate an analog reference device, while retaining interpretable and adjustable parameters. The frame duration is an important hyper-parameter of the proposed model, so an investigation was carried out into its effect on model accuracy. The optimal frame length depends on both the rate and transient decay-time of the target effect, but the frame length can be altered at inference time without a significant change in accuracy.


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
#### Citation
Accepted to the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, 4 - 7 September 2023.
```
@inproceedings{carson2023phaser,
               title={Differentiable Grey-box Modelling of Phaser Effects using Frame-based Spectral Processing},
               author={Alistair Carson and Cassia Valentini-Botinhao and Simon King and Stefan Bilbao},
               booktitle={Proceedings of the 26th International Conference on Digital Audio Effects (DAFx23)},
               year={2023},
               address={Copenhagen, Denmark}}
```