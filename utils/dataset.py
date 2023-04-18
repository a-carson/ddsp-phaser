from torch.utils.data import Dataset
import torchaudio


class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, start=0):
        if sequence_length is None:
            self._sequence_length = data["input"].shape[1]
        else:
            self._sequence_length = sequence_length
        end = start + sequence_length
        self.input_sequence = data["input"][:, start:end].view(sequence_length, 1)
        self.target_sequence = data["target"][:, start:end].view(sequence_length, 1)
        self._len = self.input_sequence.shape[1]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.input_sequence[:, index], self.target_sequence[:, index]


def load_dataset(input_path, target_path):
    input = torchaudio.load(input_path)
    target = torchaudio.load(target_path)
    sample_rate = input[1]
    assert sample_rate == target[1], "Sample rate mismatch"
    assert input[0].shape[0] == target[0].shape[0], "Signals have different lengths"
    data = {"input": input[0], "target": target[0]}
    return data, sample_rate

