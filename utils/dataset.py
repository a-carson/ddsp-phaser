from torch.utils.data import Dataset
import torchaudio
import numpy as np
import torch


class SequenceDataset(Dataset):
    def __init__(self, input, target, sequence_length, max_sequences=-1, concat_time=False):
        if sequence_length is None:
            self._sequence_length = input.shape[-1]
        else:
            self._sequence_length = sequence_length
        self.input_sequence = self.wrap_to_sequences(input, self._sequence_length, max_sequences, concat_time)
        self.target_sequence = self.wrap_to_sequences(target, self._sequence_length, max_sequences)
        self._len = self.input_sequence.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.input_sequence[index, :, :], self.target_sequence[index, :, :]

    # wraps data from  [channels, samples] -> [sequences, channels, samples]
    def wrap_to_sequences(self, data, sequence_length, max_sequences, concat_time=False):
        if max_sequences == -1:
            num_sequences = int(np.floor(data.shape[-1] / sequence_length))
        else:
            num_sequences = max_sequences
        truncated_data = data[:, 0:int(num_sequences * sequence_length)]
        wrapped_data = truncated_data.reshape((data.shape[0], num_sequences, sequence_length))

        if concat_time:
            n = torch.arange(0, sequence_length)
            time = n.repeat(1, num_sequences, 1)
            wrapped_data = torch.cat((wrapped_data, time), 0)

        return wrapped_data.permute(1, 0, 2)


def load_dataset(input_path, target_path):
    input = torchaudio.load(input_path)
    target = torchaudio.load(target_path)
    sample_rate = input[1]
    assert sample_rate == target[1], "Sample rate mismatch"
    assert input[0].shape[0] == target[0].shape[0], "Signals have different lengths"
    data = {"input": input[0], "target": target[0]}
    return data, sample_rate

