import math
from model.mlp import MLP
from model.oscillator import DampedOscillator
import torch
from torch.nn import Parameter, ParameterDict
from torch import Tensor
import torch.nn.functional as F
from torchaudio.functional import lfilter
import model.z_domain_filters as z_utils
from torchlpc import sample_wise_lpc

from typing import List, Union


def coeff_product(polynomials: Union[Tensor, List[Tensor]]) -> Tensor:
    n = len(polynomials)
    if n == 1:
        return polynomials[0]

    c1 = coeff_product(polynomials[n // 2 :])
    c2 = coeff_product(polynomials[: n // 2])
    if c1.shape[1] > c2.shape[1]:
        c1, c2 = c2, c1
    weight = c1.unsqueeze(1).flip(2)
    prod = F.conv1d(
        c2.unsqueeze(0),
        weight,
        padding=weight.shape[2] - 1,
        groups=c2.shape[0],
    ).squeeze(0)
    return prod


class PhaserSampleBased(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        hop_size,
        mlp_width=16,
        mlp_layers=3,
        mlp_activation="tanh",
        f_range=None,
        num_filters=4,
        phi=0,
    ):
        super().__init__()

        ######################
        # Fixed Parameters
        ######################
        self.K = num_filters  # num all-pass filters
        self.damped = True  # LFO damping on/off
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        ######################
        # Learnable Parameters
        ######################
        self.g1 = Parameter(Tensor([1.0]))  # through-path gain
        self.g2 = Parameter(Tensor([0.01]))  # feedback gain
        self.phi = phi
        if f_range is None:  # break-frequency max/min [Hz]
            self.depth = Parameter(0.5 * torch.rand(1))
            self.bias = Parameter(0.1 * torch.rand(1))
        else:
            d_max = Tensor([max(f_range) * 2 / sample_rate])
            d_min = Tensor([min(f_range) * 2 / sample_rate])
            self.depth = Parameter(d_max - d_min)
            self.bias = Parameter(d_min)

        ######################
        # Learnable Modules
        ######################
        self.lfo = DampedOscillator()
        self.mlp = MLP(
            width=mlp_width,
            n_hidden_layers=mlp_layers,
            activation=mlp_activation,
            bias=True,
        )
        self.filter1_params = ParameterDict(
            {
                "b": Parameter(Tensor([0.0, 0.0])),
                "a": Parameter(Tensor([0.0, 0.0])),
                "DC": Parameter(Tensor([1.0])),
            }
        )
        self.filter2_params = ParameterDict(
            {
                "b": Parameter(Tensor([0.0, 0.0])),
                "a": Parameter(Tensor([0.0, 0.0])),
            }
        )

        ################
        # for logging
        ###############
        self.max_d = 0.0
        self.min_d = 0.0

    def forward(self, x):
        device = x.device
        x = x.squeeze()

        #####################
        # shapes
        ####################
        sequence_length = x.shape[0]
        num_hops = sequence_length // self.hop_size + 2

        ###########
        # LFO
        ###########
        time = torch.arange(0, num_hops).detach().view(num_hops, 1).to(device)
        lfo = self.lfo(time, damped=self.damped)
        waveshaped_lfo = self.mlp(lfo).squeeze()

        ########################
        # Map to all-pass coeffs
        #######################
        d = self.bias + self.depth * 0.5 * (1 + waveshaped_lfo)
        p = (1.0 - torch.tan(d)) / (1.0 + torch.tan(d))

        self.max_d = torch.max(d).detach()  # for logging
        self.min_d = torch.min(d).detach()

        # upsample p
        # p = F.interpolate(
        #     p.view(1, 1, -1),
        #     size=((num_hops - 1) * self.hop_size + 1),
        #     mode="linear",
        #     align_corners=True,
        # ).view(-1)[:sequence_length]

        # filter h1
        b1 = torch.cat([self.filter1_params["DC"], self.filter1_params["b"]])
        a1 = torch.cat([b1.new_ones(1), self.filter1_params["a"]])
        h1 = lfilter(x, a1, b1, clamp=False)

        h1g = self.g1 * h1

        b2 = torch.cat([b1.new_ones(1), self.filter2_params["b"]])
        a2 = torch.cat([b1.new_ones(1), self.filter2_params["a"]])

        allpass_b = torch.stack([p, -torch.ones_like(p)], dim=1)
        allpass_a = torch.stack([torch.ones_like(p), -p], dim=1)

        combine_b = coeff_product(
            [b2.unsqueeze(0).expand(num_hops, -1)] + [allpass_b] * self.K
        )
        combine_a = coeff_product(
            [a2.unsqueeze(0).expand(num_hops, -1)] + [allpass_a] * self.K
        )

        if self.phi > 0:
            combine_denom = torch.cat(
                [combine_a, combine_a.new_zeros(num_hops, self.phi)], dim=1
            ) - self.g2.abs() * torch.cat(
                [combine_b.new_zeros(num_hops, self.phi), combine_b], dim=1
            )
        else:
            combine_denom = combine_a - self.g2.abs() * combine_b
            combine_b = combine_b / combine_denom[..., :1]
            combine_denom = combine_denom / combine_denom[..., :1]

        # upsample
        combine_b = (
            F.interpolate(
                combine_b.T.unsqueeze(0),
                size=((num_hops - 1) * self.hop_size + 1),
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .T[:sequence_length]
        )
        combine_denom = (
            F.interpolate(
                combine_denom.T.unsqueeze(0),
                size=((num_hops - 1) * self.hop_size + 1),
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .T[:sequence_length]
        )

        h1h2a = (
            F.pad(h1.view(1, 1, -1), (combine_b.shape[1] - 1, 0))
            .view(-1, 1)
            .unfold(0, combine_b.shape[1], 1)
            @ combine_b.flip(1).unsqueeze(2)
        ).squeeze()

        h1h2a = sample_wise_lpc(
            h1h2a.unsqueeze(0), combine_denom.unsqueeze(0)[..., 1:]
        ).squeeze()
        return (h1g + h1h2a).unsqueeze(0)

    def get_params(self):
        return {
            "lfo_f0": (self.sample_rate / self.hop_size)
            * self.lfo.get_omega()
            / 2
            / torch.pi,
            "lfo_r": self.lfo.get_r(),
            "lfo_phase": torch.angle(torch.view_as_complex(self.lfo.z0)),
            "lfo_max": self.max_d,
            "lfo_min": self.min_d,
            "dry_mix": self.g1.detach(),
            "feedback": self.g2.detach(),
            "delay": self.phi,
        }

    def set_frequency(self, f0):
        self.lfo.set_frequency(f0, self.sample_rate / self.hop_size)


class Phaser(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        window_length=50e-3,
        overlap_factor=0.75,
        mlp_width=16,
        mlp_layers=3,
        mlp_activation="tanh",
        f_range=None,
        num_filters=4,
        phi=None,
    ):
        super().__init__()

        ######################
        # Fixed Parameters
        ######################
        self.K = num_filters  # num all-pass filters
        self.damped = True  # LFO damping on/off
        self.sample_rate = sample_rate

        ######################
        # Init OLA
        ######################
        self.__init_OLA__(window_length, overlap_factor)

        ######################
        # Learnable Parameters
        ######################
        self.g1 = Parameter(Tensor([1.0]))  # through-path gain
        self.g2 = Parameter(Tensor([0.01]))  # feedback gain
        if phi == -1:
            self.phi = Parameter(Tensor([0.5]))  # feedback delay-line
        else:
            self.phi = Tensor([phi])
        if f_range is None:  # break-frequency max/min [Hz]
            self.depth = Parameter(0.5 * torch.rand(1))
            self.bias = Parameter(0.1 * torch.rand(1))
        else:
            d_max = Tensor([max(f_range) * 2 / sample_rate])
            d_min = Tensor([min(f_range) * 2 / sample_rate])
            self.depth = Parameter(d_max - d_min)
            self.bias = Parameter(d_min)

        ######################
        # Learnable Modules
        ######################
        self.lfo = DampedOscillator()
        self.mlp = MLP(
            width=mlp_width,
            n_hidden_layers=mlp_layers,
            activation=mlp_activation,
            bias=True,
        )
        self.filter1 = z_utils.Biquad(Nfft=self.Nfft, normalise=False)
        self.filter2 = z_utils.Biquad(Nfft=self.Nfft, normalise=True)

        ################
        # for logging
        ###############
        self.max_d = 0.0
        self.min_d = 0.0

    def __init_OLA__(self, window_length, overlap_factor):
        self.overlap = overlap_factor
        hops_per_frame = int(1 / (1 - self.overlap))
        self.window_size = hops_per_frame * math.floor(
            window_length * self.sample_rate / hops_per_frame
        )  # ensure constant OLA
        self.hop_size = int(self.window_size / hops_per_frame)
        self.Nfft = 2 ** math.ceil(math.log2(self.window_size) + 1)
        self.register_buffer(
            "window_idx", torch.arange(0, self.window_size, 1).detach()
        )
        self.register_buffer("hann", z_utils.hann_window(self.window_size).detach())
        self.register_buffer("z", z_utils.z_inverse(self.Nfft, full=False).detach())
        self.OLA_gain = (3 / 8) * (self.window_size / self.hop_size)

    def forward(self, x):
        device = x.device

        #####################
        # shapes
        ####################
        sequence_length = x.shape[1]
        num_hops = math.floor(sequence_length / self.hop_size) + 1

        ###########
        # LFO
        ###########
        time = torch.arange(0, num_hops).detach().view(num_hops, 1).to(device)
        lfo = self.lfo(time, damped=self.damped)
        waveshaped_lfo = self.mlp(lfo)

        ########################
        # Map to all-pass coeffs
        #######################
        d = self.bias + self.depth * 0.5 * (1 + waveshaped_lfo)
        p = (1.0 - torch.tan(d)) / (1.0 + torch.tan(d))
        ap_params = p.to(device)
        self.max_d = torch.max(d).detach()  # for logging
        self.min_d = torch.min(d).detach()

        #########################
        # STFT approximation
        ########################
        X = torch.stft(
            x,
            n_fft=self.Nfft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            return_complex=True,
            onesided=True,
            center=True,
            pad_mode="constant",
            window=self.hann,
        )
        Y = X * self.transfer_matrix(ap_params).permute(1, 0).unsqueeze(0)
        return torch.istft(
            Y,
            n_fft=self.Nfft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            window=self.hann,
            center=True,
            length=x.shape[1],
        )

    def transfer_matrix(self, p):
        h1 = self.filter1()
        h2 = self.filter2()
        a = torch.pow(((p - self.z) / (1 - p * self.z)), self.K)
        denom = (
            1 - torch.pow(self.z, torch.relu(self.phi)) * torch.abs(self.g2) * h2 * a
        )
        out = h1 * (self.g1 + h2 * a / denom)
        return out

    def get_params(self):
        return {
            "lfo_f0": (self.sample_rate / self.hop_size)
            * self.lfo.get_omega()
            / 2
            / torch.pi,
            "lfo_r": self.lfo.get_r(),
            "lfo_phase": torch.angle(torch.view_as_complex(self.lfo.z0)),
            "lfo_max": self.max_d,
            "lfo_min": self.min_d,
            "dry_mix": self.g1.detach(),
            "feedback": self.g2.detach(),
            "delay": self.phi.detach(),
        }

    def set_frequency(self, f0):
        self.lfo.set_frequency(f0, self.sample_rate / self.hop_size)

    def set_window_size(self, window_length):
        lfo_freq = self.get_params()["lfo_f0"]
        self.__init_OLA__(window_length=window_length, overlap_factor=self.overlap)
        self.set_frequency(lfo_freq)
        self.filter1.set_Nfft(Nfft=self.Nfft)
        self.filter2.set_Nfft(Nfft=self.Nfft)
