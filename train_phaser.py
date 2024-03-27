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
from model.digital_phaser import DigitalPhaser
import matplotlib.pyplot as plt


class Phaser(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = phaser.Phaser(
            sample_rate=args.sample_rate,
            window_length=args.window_length,
            overlap_factor=args.overlap,
            mlp_activation=args.mlp_activation,
            mlp_width=args.mlp_width,
            mlp_layers=args.mlp_layers,
            phi=args.phi,
            )

        self.save_hyperparameters()
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
        self.wandb = args.wandb
        self.sample_based = args.exact
        self.lr = args.lr
        self.automatic_optimization = False

    def forward(self, x):
        if self.sample_based:
            return self.model.forward_sample_based(x)
        else:
            return self.model.forward(x)

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        # Training
        x, y = batch
        y_pred, _ = self(x)
        loss = self.loss_fcn(y.squeeze(1), y_pred)

        # optimize
        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradients
        if self.sample_based:
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()

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
        x, y = batch
        self.model.damped = False
        y_hat, p = self.model.forward(x)
        y_hat_sample_base, _ = self.model.forward_sample_based(x)
        self.model.damped = True
        loss_esr = self.esr(y.squeeze(1), y_hat)
        loss_esr_sample_based = self.esr(y, y_hat_sample_base.squeeze(1))
        self.log("val_loss_frame", loss_esr, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss_sample", loss_esr_sample_based, on_epoch=True, prog_bar=False, logger=True)

        audio = torch.flatten(y_hat)

        if self.wandb:
            if self.current_epoch % 250 == 0:
                wandb.log(
                    {'Audio/' + "Val": wandb.Audio(audio.cpu().detach().numpy(), caption="Val", sample_rate=44100),
                     'epoch': self.current_epoch})

                H1 = self.model.filter1().detach().cpu()
                plt.plot(20 * torch.log10(H1.abs()))
                wandb.log({"|H|": plt})

                h1 = torch.fft.irfft(H1)
                plt.plot(h1)
                wandb.log({"h[n]": plt})

                plt.plot(p.flatten().detach().cpu())
                wandb.log({"LFO": plt})

        if self.current_epoch == 0:
            audio = torch.flatten(y)
            if self.wandb:
                wandb.log(
                    {'Audio/' + "Target": wandb.Audio(audio.cpu().detach().numpy(), caption="Val", sample_rate=44100),
                     'epoch': self.current_epoch})
        return y_hat

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.model.damped = False
        y_hat, _ = self.model.forward(x)
        y_hat_sample_base, _ = self.model.forward_sample_based(x)
        self.model.damped = True
        loss_esr = self.esr(y.squeeze(1), y_hat)
        loss_esr_sample_based = self.esr(y, y_hat_sample_base.squeeze(1))
        self.log("test_loss_frame", loss_esr, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_loss_sample", loss_esr_sample_based, on_epoch=True, prog_bar=False, logger=True)
        audio = torch.flatten(y_hat)
        if self.wandb:
            wandb.log(
                {'Audio/' + "Test": wandb.Audio(audio.cpu().detach().numpy(), caption="Test", sample_rate=44100),
                 'epoch': self.current_epoch})

        return y_hat

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.lr, eps=1e-08, weight_decay=0)
        return opt

    def on_train_epoch_start(self):
        self.last_time = time.time()

LOG_PATH = os.environ.get('WANDB_LOG_DIR', default='wandb_logs/')



if __name__ == "__main__":
    # INPUT ARGUMENTS ------------------------------
    parser = ArgumentParser()

    # general
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--log_path", type=str, default="./out")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=50)


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
    parser.add_argument("--manual_seed", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-4)


    # data
    parser.add_argument(
        "--dataset_input", type=str, default="audio_data/small_stone/input_dry.wav"
    )
    parser.add_argument(
        "--dataset_target",
        type=str,
        default="audio_data/small_stone/colour=0_rate=3oclock.wav",
    )
    parser.add_argument("--train_data_length", type=float, default=5.67)
    parser.add_argument("--sequence_length_test", type=float, default=10.0)

    parser.add_argument("--synthetic_data", action=argparse.BooleanOptionalAction)
    parser.add_argument("--target_f0", type=float, default=1.0)


    args = parser.parse_args()
    if args.manual_seed >= 0:
        torch.manual_seed(args.manual_seed)

    if torch.cuda.is_available():
        pin_memory = True
        num_workers = 0
    else:
        pin_memory = False
        num_workers = 0

    # LOAD DATA -------------------
    audio_data, sample_rate = ds.load_dataset(args.dataset_input, args.dataset_target)
    args.sample_rate = sample_rate

    audio_data["input"] = torchaudio.functional.highpass_biquad(audio_data["input"], cutoff_freq=20, sample_rate=sample_rate)
    audio_data["target"] = torchaudio.functional.highpass_biquad(audio_data["target"], cutoff_freq=20, sample_rate=sample_rate)
    audio_data["input"] = audio_data["input"].double()
    audio_data["target"] = audio_data["target"].double()


    if args.synthetic_data:
        print("Generating synthetic data...")
        dp = DigitalPhaser(sample_rate=args.sample_rate, f0=args.target_f0)
        audio_data['target'] = torch.from_numpy(dp(audio_data["input"].numpy()))
        print('Generating synthetic data done.')

    train_data_length = int(args.train_data_length * sample_rate)
    test_seq_length = int(args.sequence_length_test * sample_rate)
    train_end = int(60 * sample_rate)
    train_start = train_end - train_data_length  # custom dataset contains 60s of chirp signal (training data) followed by test audio


    train_loader = DataLoader(
        dataset=ds.SequenceDataset(
            input=audio_data["input"][..., train_start:train_end],
            target=audio_data["target"][..., train_start:train_end],
            sequence_length=train_data_length,
        ),
        pin_memory=pin_memory,
        num_workers=num_workers)
    test_loader = DataLoader(
        dataset=ds.SequenceDataset(
            input=audio_data["input"][..., train_start:],
            target=audio_data["target"][..., train_start:],
            sequence_length=test_seq_length + train_data_length,
            max_sequences=1,
        ))  # must be same as train loader for LFO phase consistency

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
        model = Phaser.load_from_checkpoint(args.checkpoint_path,
            args=args,
            sample_rate=sample_rate)

    model.double()

    # freezes osc parameters
    if args.freeze != 0:
        model.model.set_frequency(args.f0)
        model.model.damped = False
        model.model.lfo.osc.z.requires_grad = False
        model.model.lfo.osc.z0.requires_grad = False

    # optional wandb logger
    if args.wandb is not None:
        path = os.path.join(LOG_PATH, args.project_name)
        if not os.path.exists(path):
            os.mkdir(path)
        wandb_logger = WandbLogger(project=args.project_name,
                                   name=args.run_name,
                                   group=args.experiment_name,
                                   save_dir=path)
    else:
        wandb_logger = None

    # TRAIN! ---------------------------
    trainer = pl.Trainer(
        log_every_n_steps=10,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss_sample", filename="{epoch}-{val_loss_sample:.4f}", save_last=True)],
        check_val_every_n_epoch=args.check_val_every_n_epoch
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')

    # # SAVE AUDIO -----------------------
    # model.eval()
    # with torch.no_grad():
    #     model.model.damped = False
    #     out = model.model.forward(audio_data["input"])
    # path = "audio_out.wav"
    # if wandb_logger is not None:
    #     path = os.path.join(args.project_name, wandb_logger.version, path)
    # torchaudio.save(path, out.detach(), sample_rate)
    # if wandb_logger is not None:
    #     wandb.save(path)
    # print("Saved")
