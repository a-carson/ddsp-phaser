import torch
from torch.nn import Parameter


class DampedOscillator(torch.nn.Module):

    default_damping = 0.7
    default_amplitude = 1.0

    def __init__(self):
        super().__init__()

        omega = 0.1 * torch.randn(1)
        w = torch.tensor([torch.sqrt(-torch.log(torch.Tensor([self.default_damping]))), omega])
        self.w = Parameter(w)

        phi = torch.randn(1)
        z0 = torch.polar(torch.Tensor([self.default_amplitude]), phi)
        self.z0 = Parameter(torch.view_as_real(z0))



    def forward(self, n, damped, normalise=False):

        z = torch.polar(self.get_r(), self.get_omega())
        z0 = torch.view_as_complex(self.z0)

        if not damped:
            z = z / torch.abs(z)

        if normalise:
            z0 = z0 / torch.abs(z0)

        return torch.real(z0 * z ** n)


    def set_omega(self, omega):
        w = torch.tensor([torch.sqrt(-torch.log(torch.Tensor([self.default_damping]))), omega])
        self.w = Parameter(w)

    def set_phase(self, phi):
        z0 = torch.polar(torch.Tensor([self.default_amplitude]), torch.Tensor([phi]))
        self.z0 = Parameter(torch.view_as_real(z0))

    def get_omega(self):
        return self.w[1]

    def get_r(self):
        return torch.exp(-self.w[0] ** 2)

    def set_frequency(self, f0, sample_rate):
        omega = 2 * torch.pi * f0 / sample_rate
        self.set_omega(omega)

