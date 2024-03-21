import torch
from scipy import signal
import numpy as np
import torchaudio

class DigitalPhaser(torch.nn.Module):
    def __init__(self, sample_rate, f0, wet_mix=1.0, f_min=1e3, f_max=4e3, feedback=0.7):
        super().__init__()
        self.sample_rate = sample_rate
        self.f0 = f0
        self.d_min = f_min * 2 / sample_rate
        self.d_max = f_max * 2 / sample_rate
        self.depth = (self.d_max - self.d_min)*0.5
        self.wet_mix = wet_mix
        self.feedback = feedback

    def forward(self, x):
        t = np.arange(0, x.shape[-1]) / self.sample_rate
        lfo = self.d_min + self.depth * (1.0 + signal.sawtooth(2 * np.pi * self.f0 * t, 0.5))
        p = (1 - np.tan(lfo)) / (1 + np.tan(lfo))
        y = all_pass_chain(x, p, self.wet_mix, self.feedback)
        return y


def all_pass_chain(x, p, wet_mix, feedback):
    # init
    y = np.zeros_like(x)

    if p.shape[-1] == 1:
        p = np.ones_like(x) * p

    # states
    h1_prev = 0
    x1_prev = 0
    h2_prev = 0
    x2_prev = 0
    h3_prev = 0
    x3_prev = 0
    h4_prev = 0
    x4_prev = 0

    # DSP loop
    for n in range(x.shape[-1]):
        # read input
        x1 = x[..., n] + h4_prev * feedback

        # AP1
        h1 = p[n] * (x1 + h1_prev) - x1_prev
        x2 = h1
        # AP2
        h2 = p[n] * (x2 + h2_prev) - x2_prev
        x3 = h2

        # AP1
        h3 = p[n] * (x3 + h3_prev) - x3_prev
        x4 = h3
        # AP2
        h4 = p[n] * (x4 + h4_prev) - x4_prev

        # output
        y[..., n] = h4

        # update states
        h1_prev = h1
        x1_prev = x1
        h2_prev = h2
        x2_prev = x2
        h3_prev = h3
        x3_prev = x3
        h4_prev = h4
        x4_prev = x4

    y += wet_mix * x
    return y

if __name__ == '__main__':
    x, sample_rate = torchaudio.load('../audio_data/digital_phaser/input_dry.wav')
    dp = DigitalPhaser(sample_rate=sample_rate, f0=1.0)
    y = dp(x.numpy())
    torchaudio.save('../audio_data/digital_phaser/dp_out.wav', torch.from_numpy(y), sample_rate)
    print(y)
