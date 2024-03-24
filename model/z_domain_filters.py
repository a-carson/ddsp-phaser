import torch
from torch import Tensor as T
from torch.nn import Parameter

def calc_lp_biquad_coeff(w: T, q: T, eps: float = 1e-3) -> (T, T):
    assert w.ndim == 2
    assert q.ndim == 2
    assert 0.0 <= w.min()
    assert torch.pi >= w.max()
    assert 0.0 < q.min()

    stability_factor = 1.0 - eps
    alpha_q = torch.sin(w) / (2 * q)
    a0 = 1.0 + alpha_q
    a1 = -2.0 * torch.cos(w) * stability_factor
    a1 = a1 / a0
    a2 = (1.0 - alpha_q) * stability_factor
    a2 = a2 / a0
    assert (a1.abs() < 2.0).all()
    assert (a2 < 1.0).all()
    assert (a1 < a2 + 1.0).all()
    assert (a1 > -(a2 + 1.0)).all()
    a = torch.stack([a1, a2], dim=2)

    b0 = (1.0 - torch.cos(w)) / 2.0
    b0 = b0 / a0
    b1 = 1.0 - torch.cos(w)
    b1 = b1 / a0
    b2 = (1.0 - torch.cos(w)) / 2.0
    b2 = b2 / a0
    b = torch.stack([b0, b1, b2], dim=2)

    return a, b

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
        self.ff_params = Parameter(T([0.0, 0.0]))
        self.fb_params = Parameter(T([0.0, 0.0]))
        self.DC = Parameter(T([1.0]))
        self.register_buffer("pows", T([1.0, 2.0]))
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

class LPBiquad(torch.nn.Module):
    def __init__(self, Nfft, normalise=False):
        super().__init__()
        self.omega = Parameter(T([torch.pi]))
        self.q = Parameter(T([0.707]))
        self.register_buffer("pows", T([1.0, 2.0]))
        self.register_buffer("z", z_inverse(Nfft, full=False).detach().unsqueeze(1))
        self.register_buffer("zpows", torch.pow(self.z, self.pows))
        self.normalise = normalise
        self.Nfft = Nfft
    
    def forward(self):
        a, b = calc_lp_biquad_coeff(self.omega.view(1, -1), self.q.view(1, -1))
        ff = torch.sum(b.flatten()[1:] * self.zpows, 1)
        ff += b.flatten()[0]
        fb = 1.0 + torch.sum(a.flatten() * self.zpows, 1)
        return ff / fb
    
    def set_Nfft(self, Nfft):
        self.Nfft = Nfft
        self.register_buffer(
            "z", z_inverse(self.Nfft, full=False).detach().unsqueeze(1)
        )
        self.register_buffer("zpows", torch.pow(self.z, self.pows))


class SVFBiquad(torch.nn.Module):
    def __init__(self, Nfft, normalise=False):
        super().__init__()
        self.omega = Parameter(T([torch.pi]))
        self.R = Parameter(T([0.707]))
        self.gain = Parameter(T([1.0]))
        self.register_buffer("pows", T([1.0, 2.0]))
        self.register_buffer("z", z_inverse(Nfft, full=False).detach().unsqueeze(1))
        self.register_buffer("zpows", torch.pow(self.z, self.pows))
        self.normalise = normalise
        self.Nfft = Nfft

    def forward(self):
        g = torch.tan(self.omega/2)
        num = g**2 * self.z**2
        num += 2*g**2 * self.z
        num += g**2

        denom = (1 + g**2 - 2*self.R)*self.z**2
        denom += (2*g**2 - 2) * self.z
        denom += 1.0 + g**2 + 2*self.R*g

        return self.gain * (num / denom).squeeze()

    def set_Nfft(self, Nfft):
        self.Nfft = Nfft
        self.register_buffer(
            "z", z_inverse(self.Nfft, full=False).detach().unsqueeze(1)
        )
        self.register_buffer("zpows", torch.pow(self.z, self.pows))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bq = SVFBiquad(Nfft=1024)
    plt.plot(10 * torch.log10(bq.forward().abs()).detach().numpy())
    plt.show()
