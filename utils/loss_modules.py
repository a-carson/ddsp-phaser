import torch
from torch.nn import functional


class ESRLoss(torch.nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, target, predicted):
        mse = torch.mean(torch.square(torch.subtract(target, predicted)))
        signal_energy = torch.mean(torch.square(target))
        return torch.div(mse, signal_energy + self.epsilon)


"""
Spectral loss
"""


class SpectralLoss(torch.nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int):
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.epsilon = 1e-8

    # requires shapes (N, L)
    def forward(self, target, predicted):
        stft_target = torch.abs(
            torch.stft(
                target,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                return_complex=True,
            )
        )
        stft_pred = torch.abs(
            torch.stft(
                predicted,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                return_complex=True,
            )
        )

        convergence_loss = torch.norm(stft_target - stft_pred) / (
            torch.norm(stft_target) + self.epsilon
        )
        magnitude_loss = torch.norm(
            torch.log10(stft_target + self.epsilon)
            - torch.log10(stft_pred + self.epsilon),
            p=1,
        ) / torch.numel(stft_target)
        return convergence_loss + magnitude_loss


"""
Multi Resolution Spectral Loss module
"""


class MRSL(torch.nn.Module):
    def __init__(self, fft_lengths=None, window_sizes=None, overlap=0.25):
        super(MRSL, self).__init__()

        if fft_lengths is None:
            fft_lengths = [4096, 2048, 1024]
        if window_sizes is None:
            window_sizes = [int(0.5 * n) for n in fft_lengths]

        hop_sizes = [int(w * overlap) for w in window_sizes]

        assert [
            len(fft_lengths) == len(window_sizes),
            "window_sizes and fft_lengths must be the same length",
        ]

        self.spec_losses = torch.nn.ModuleList()
        for n_fft, n_win, hop in zip(fft_lengths, window_sizes, hop_sizes):
            self.spec_losses.append(
                SpectralLoss(n_fft=n_fft, win_length=n_win, hop_length=hop)
            )

    def forward(self, target, predicted):
        return sum(map(lambda f: f(target, predicted), self.spec_losses)) / len(
            self.spec_losses
        )


def pre_emphasis_hpf(signal):
    zero_padded = functional.pad(signal, (1, 1), "constant", 0)
    return zero_padded[1:] - 0.85 * zero_padded[:-1]
