import math
from model.mlp import MLP
from model.oscillator import DampedOscillator
import torch
from torch.nn import Parameter
from torch import Tensor
import model.z_domain_filters as z_utils


class Phaser(torch.nn.Module):

    def __init__(self, sample_rate,
                 window_length=50e-3,
                 overlap_factor=0.75,
                 mlp_width=16,
                 mlp_layers=3,
                 mlp_activation='tanh',
                 f_range=None,
                 num_filters=4):

        super().__init__()

        ######################
        # Fixed Parameters
        ######################
        self.K = num_filters                                    # num all-pass filters
        self.damped = True                                      # LFO damping on/off
        self.sample_rate = sample_rate

        ######################
        # Init OLA
        ######################
        self.__init_OLA__(window_length, overlap_factor)

        ######################
        # Learnable Parameters
        ######################
        self.g1 = Parameter(Tensor([1.0]))                      # through-path gain
        self.g2 = Parameter(Tensor([0.01]))                     # feedback gain
        self.phi = Parameter(Tensor([0.5]))                     # feedback delay-line
        if f_range is None:                                     # break-frequency max/min [Hz]
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
        self.mlp = MLP(width=mlp_width, n_hidden_layers=mlp_layers, activation=mlp_activation, bias=True)
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
        self.window_size = hops_per_frame * math.floor(window_length * self.sample_rate / hops_per_frame) # ensure constant OLA
        self.hop_size = int(self.window_size / hops_per_frame)
        self.Nfft = 2 ** math.ceil(math.log2(self.window_size))
        self.window_idx = torch.arange(0, self.window_size, 1).detach()
        self.hann = z_utils.hann_window(self.window_size).detach()
        self.z = z_utils.z_inverse(self.Nfft, full=False).detach()
        self.OLA_gain = (3/8)*(self.window_size / self.hop_size)

    def forward(self, x):

        device = x.device

        #####################
        # shapes
        ####################
        sequence_length = x.shape[1]
        N = self.window_size                # window size
        Nfft = self.Nfft                    # fft length
        Nfft_half = int(Nfft / 2)
        num_hops = math.ceil(sequence_length / self.hop_size)

        ##############
        # init vectors
        ##############
        Npad = int(N/2)
        x_pad = torch.nn.functional.pad(x, (Npad, Npad), 'constant', 0)
        output_sequence = torch.zeros(x_pad.shape).to(device)

        ###########
        # LFO
        ###########
        time = torch.arange(0, num_hops).detach().view(num_hops, 1)
        lfo = self.lfo(time, damped=self.damped)
        waveshaped_lfo = self.mlp(lfo)

        ########################
        # Map to all-pass coeffs
        #######################
        d = self.bias + self.depth * 0.5 * (1 + waveshaped_lfo)
        p = (1.0 - torch.tan(d)) / (1.0 + torch.tan(d))
        ap_params = p.to(device)
        self.max_d = torch.max(d).detach()        # for logging
        self.min_d = torch.min(d).detach()

        ##########################
        # DSP loop
        #########################
        for m in range(num_hops):

            # window and take FFT
            x_frame = x_pad[:, self.window_idx + m * self.hop_size]
            x_dft = torch.fft.fft(self.hann * x_frame, Nfft)[:, 0:Nfft_half+1]

            # manipulate DFT
            y_dft = x_dft * self.transfer_function(ap_params[m])
            y_symm = z_utils.make_hermetian(y_dft, dim=1)

            # overlap-add back into output
            y_frame = torch.real(torch.fft.ifft(y_symm))
            output_sequence[:, self.window_idx + m*self.hop_size] += self.hann * y_frame[:, :N]

        return output_sequence[:, Npad:-Npad] / self.OLA_gain

    def transfer_function(self, pm):
        h1 = self.filter1()
        h2 = self.filter2()
        a = torch.pow(((pm - self.z) / (1 - pm * self.z)), self.K)
        denom = 1 - torch.pow(self.z, torch.relu(self.phi)) * torch.abs(self.g2) * h2 * a
        return h1 * (self.g1 + h2 * a / denom)

    def get_params(self):
        return {
                "lfo_f0": (self.sample_rate / self.hop_size) * torch.angle(torch.view_as_complex(self.lfo.za)) / 2 / torch.pi,
                "lfo_r": torch.norm(self.lfo.za),
                "lfo_phase": torch.angle(torch.view_as_complex(self.lfo.zb)),
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






