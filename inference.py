from argparse import ArgumentParser
import torch
import torchaudio
from train_phaser import Phaser
from utils import dataset as ds


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/ss-A.ckpt')
    parser.add_argument("--dataset_input", type=str, default='audio_data/small_stone/input_dry.wav')
    parser.add_argument("--dataset_target", type=str, default='audio_data/small_stone/colour=0_rate=3oclock.wav')
    parser.add_argument("--window_length", type=float, default=0.08)
    parser.add_argument("--f0", type=float, default=0.0)
    args = parser.parse_args()

    # LOAD DATA
    data, sample_rate = ds.load_dataset(args.dataset_input, args.dataset_target)

    # LOAD MODEL
    state_dict = torch.load(args.checkpoint_path)
    model = Phaser.load_from_checkpoint(args.checkpoint_path, sample_rate=sample_rate, strict=False)

    # CHANGE PARAMETERS (optional)
    model.model.set_window_size(args.window_length)
    model.model.damped = False
    if args.f0 > 0.0:
        model.model.set_frequency(args.f0)

    # RUN
    model.eval()
    with torch.no_grad():
        out = model.forward(data["input"])

    # SAVE
    torchaudio.save("inference_out.wav", out.detach(), sample_rate)
    print("Saved")



