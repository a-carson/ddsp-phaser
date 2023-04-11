import torch
from torch.nn import Parameter


class DampedOscillator(torch.nn.Module):

    default_damping = 0.7

    def __init__(self,
                 omega=None,
                 phi=None):
        super().__init__()

        self.za = None
        if omega is None:
            omega = torch.pi * torch.rand(1)    # random frequency
        self.set_omega(omega)

        if phi is None:
            zb = torch.tensor([1.0, 0])
        else:
            zb = torch.tensor([torch.cos(torch.tensor(phi)),
                               torch.sin(torch.tensor(phi))])
        self.zb = Parameter(zb)

    def forward(self, n, damped, normalise=False):

        za = torch.view_as_complex(self.za)
        zb = torch.view_as_complex(self.zb)

        if not damped:
            za = za / torch.abs(za)

        if normalise:
            zb = zb / torch.abs(zb)

        x = torch.real(zb * za ** n)
        return x

    def set_omega(self, omega):
        za = torch.tensor([torch.cos(torch.Tensor([omega])), torch.sin(torch.Tensor([omega]))])
        self.za = Parameter(self.default_damping * za)

    def get_omega(self):
        return torch.angle(torch.view_as_complex(self.za))

    def set_frequency(self, f0, sample_rate):
        omega = 2 * torch.pi * f0 / sample_rate
        self.set_omega(omega)




