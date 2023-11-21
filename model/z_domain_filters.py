import torch
from torch import Tensor
from torch.nn import Parameter


def z_inverse(num_dft_bins, full=False):
    if full:
        n = torch.arange(0, num_dft_bins, 1)
    else:
        n = torch.arange(0, int(num_dft_bins / 2) + 1, 1)

    omega = 2 * torch.pi * n / num_dft_bins
    real = torch.cos(omega)
    imag = -torch.sin(omega)
    return torch.view_as_complex(torch.stack((real, imag), 1))


def hann_window(N):
    n = torch.arange(0, N, 1)
    return 0.5 * (1 - torch.cos(2 * torch.pi * n / N))


def make_hermetian(x, dim):
    length = x.shape[dim]
    interior = torch.index_select(x, dim, torch.arange(1, length - 1, 1))
    return torch.cat((x, torch.flip(torch.conj(interior), dims=[dim])), dim)


class Biquad(torch.nn.Module):
    def __init__(self, Nfft, normalise=False):
        super().__init__()
        self.ff_params = Parameter(Tensor([0.0, 0.0]))
        self.fb_params = Parameter(Tensor([0.0, 0.0]))
        self.DC = Parameter(Tensor([1.0]))
        self.register_buffer("pows", Tensor([1.0, 2.0]))
        self.register_buffer("z", z_inverse(Nfft, full=False).detach().unsqueeze(1))
        self.register_buffer("zpows", torch.pow(self.z, self.pows))
        self.normalise = normalise
        self.Nfft = Nfft

    def forward(self):
        ff = torch.sum(self.ff_params * self.zpows, 1)
        if self.normalise:
            ff += 1.0
        else:
            ff += self.DC
        fb = 1.0 + torch.sum(self.fb_params * self.zpows, 1)
        return ff / fb

    def set_Nfft(self, Nfft):
        self.Nfft = Nfft
        self.register_buffer(
            "z", z_inverse(self.Nfft, full=False).detach().unsqueeze(1)
        )
        self.register_buffer("zpows", torch.pow(self.z, self.pows))
