import argparse
from argparse import ArgumentParser
from utils import dataset as ds
from utils import loss_modules
from model import phaser
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import time
import torch
from torch.utils.data import DataLoader
import torchaudio
import wandb


class Phaser(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if not args.exact:
            self.model = phaser.Phaser(
                sample_rate=args.sample_rate,
                window_length=args.window_length,
                overlap_factor=args.overlap,
                mlp_activation=args.mlp_activation,
                mlp_width=args.mlp_width,
                mlp_layers=args.mlp_layers,
            )
        else:
            self.model = phaser.PhaserSampleBased(
                sample_rate=args.sample_rate,
                hop_size=int(
                    args.window_length * args.sample_rate * (1 - args.overlap)
                ),
                mlp_activation=args.mlp_activation,
                mlp_width=args.mlp_width,
                mlp_layers=args.mlp_layers,
                phi=args.phi,
            )

        self.save_hyperparameters()
        self.train_sequence_length = int(args.sequence_length * args.sample_rate)
        self.esr = loss_modules.ESRLoss()
        self.mrsl = loss_modules.MRSL(fft_lengths=[512, 1024, 2048])
        if args.loss_fcn == "esr":
            self.loss_fcn = self.esr
            print("Loss function: ESR")
        else:
            self.loss_fcn = self.mrsl
            print("Loss function: MRSL")
        self.last_time = time.time()
        self.epoch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training
        x, y = batch
        y_pred = self.model.forward(x)
        loss = self.loss_fcn(y, y_pred)

        # Logging
        self.log("train_loss_esr", loss, on_step=True, prog_bar=True, logger=True)
        new_time = time.time()
        self.log(
            "time_per",
            new_time - self.last_time,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        params = model.model.get_params()
        for key, value in params.items():
            self.log(key, value, on_step=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.epoch += 1
        if self.epoch % 50 == 0:
            return self.test_step(batch, batch_idx)
        else:
            return

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.model.damped = False
        y_hat = self.model.forward(x)
        self.model.damped = True
        loss_esr = self.esr(y, y_hat)
        loss_mrsl = self.mrsl(y, y_hat)
        self.log("test_loss_esr", loss_esr, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            "test_loss_mrsl", loss_mrsl, on_epoch=True, prog_bar=False, logger=True
        )
        return y_hat

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-08, weight_decay=0)
        return opt

    def on_train_epoch_start(self):
        self.last_time = time.time()


if __name__ == "__main__":
    # INPUT ARGUMENTS ------------------------------
    parser = ArgumentParser()

    # general
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_path", type=str, default="./out")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_epochs", type=int, default=5000)

    # model
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--phi", type=int, default=0)
    parser.add_argument("--f0", type=float, default=0.0)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--window_length", type=float, default=0.08)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--loss_fcn", type=str, default="esr")
    parser.add_argument("--mlp_activation", type=str, default="tanh")
    parser.add_argument("--mlp_width", type=int, default=8)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--manual_seed", action=argparse.BooleanOptionalAction)

    # data
    parser.add_argument(
        "--dataset_input", type=str, default="audio_data/small_stone/input_dry.wav"
    )
    parser.add_argument(
        "--dataset_target",
        type=str,
        default="audio_data/small_stone/colour=0_rate=3oclock.wav",
    )
    parser.add_argument("--sequence_length", type=float, default=2.67)
    parser.add_argument("--sequence_length_test", type=float, default=10.0)

    args = parser.parse_args()
    if args.manual_seed is not None:
        torch.manual_seed(0)

    if torch.cuda.is_available():
        pin_memory = True
        num_workers = 0
    else:
        pin_memory = False
        num_workers = 0

    # LOAD DATA -------------------
    audio_data, sample_rate = ds.load_dataset(args.dataset_input, args.dataset_target)
    args.sample_rate = sample_rate

    train_seq_length = int(args.sequence_length * sample_rate)
    test_seq_length = int(args.sequence_length_test * sample_rate)
    start = (
        int(60 * sample_rate) - train_seq_length
    )  # custom dataset contains 60s of chirp signal (training data) followed by test audio

    train_loader = DataLoader(
        dataset=ds.SequenceDataset(
            data=audio_data, sequence_length=train_seq_length, start=start
        ),
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=ds.SequenceDataset(
            data=audio_data,
            sequence_length=test_seq_length + train_seq_length,
            start=start,
        )
    )  # must be same as train loader for LFO phase consistency

    # LOAD MODEL --------------------
    if args.checkpoint_path == "":
        model = Phaser(args=args)
        if args.f0 == 0.0:
            torch.manual_seed(int(time.time()))  # seed rand gen
            rand_f0 = 0.1 * torch.randn(1)  # random init frequency in Hz
            print("Init f0: ", rand_f0)
            model.model.set_frequency(rand_f0)
        else:
            model.model.set_frequency(args.f0)
    else:
        model = Phaser.load_from_checkpoint(
            args.checkpoint_path, sample_rate=sample_rate
        )

    # freezes osc parameters
    if args.freeze != 0:
        model.model.set_frequency(args.f0)
        model.model.damped = False
        model.model.lfo.osc.z.requires_grad = False
        model.model.lfo.osc.z0.requires_grad = False

    # optional wandb logger
    if args.wandb is not None:
        wandb_logger = WandbLogger(project=args.project_name, name=args.run_name)
    else:
        wandb_logger = None

    # TRAIN! ---------------------------
    trainer = pl.Trainer(
        log_every_n_steps=10,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(dataloaders=test_loader)

    # SAVE AUDIO -----------------------
    model.eval()
    with torch.no_grad():
        model.model.damped = False
        out = model.model.forward(audio_data["input"])
    path = "audio_out.wav"
    if wandb_logger is not None:
        path = os.path.join(args.project_name, wandb_logger.version, path)
    torchaudio.save(path, out.detach(), sample_rate)
    if wandb_logger is not None:
        wandb.save(path)
    print("Saved")
