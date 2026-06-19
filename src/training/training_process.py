import json
import os

import torch
import torchaudio
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
from transformers import VitsTokenizer

from debugger import Debugger
from src.VitsAudioDataset import VitsAudioDataset, create_data_loader
from src.model import SimpleVITS
from src.training.config import TrainingConfig
from src.training.step import train_step


def create_training_config():
    tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-rus")
    vocab_size = len(tokenizer)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = SimpleVITS(vocab_size).to(device)

    # Separate the optimizers as your code requires
    optimizer_g = torch.optim.AdamW(model.generator.parameters(), lr=1e-5, betas=(0.5, 0.9))
    optimizer_d = torch.optim.AdamW(model.discriminator.parameters(), lr=5e-5, betas=(0.5, 0.9))

    # def lr_lambda(step):
    #     if step < 1000:
    #         return step / 1000  # Warmup from 0 to 1 over 1000 steps
    #     return 1.0
    #
    # scheduler_g = LambdaLR(optimizer_g, lr_lambda=lr_lambda)
    # scheduler_d = LambdaLR(optimizer_d, lr_lambda=lr_lambda)

    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        # number of digital samples analyzed in a single window of time
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=80,
        f_min=0,
        f_max=8000,
    ).to(device)

    debugger = Debugger()

    return TrainingConfig(
        tokenizer=tokenizer,
        debugger=debugger,
        vocab_size=len(tokenizer),
        model=model,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        # scheduler_g=scheduler_g,
        # scheduler_d=scheduler_d,
        mel_extractor=mel_extractor,
        device=device,
        min_audio_length=1024,
        mel_hop_length=512,
        num_epochs=1000,
        save_every=100,
    )


def perform_training(config: TrainingConfig):
    config.model.train()
    global_step = 0

    with open("sonya_dataset/index.json", 'r', encoding='utf-8') as f:
        index = json.load(f)
        data = [(audio_path, index[audio_path]) for audio_path in index]

    dataset = VitsAudioDataset(data, config.tokenizer, config.mel_extractor, config.device, config.min_audio_length)
    data_loader = create_data_loader(dataset)

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")

    for epoch in range(config.num_epochs):
        progres_bar = tqdm(data_loader, desc="Epoch {}".format(epoch + 1))

        g_loss = SmallDataAggregator()
        d_loss = SmallDataAggregator()
        mel_loss = SmallDataAggregator()
        dur_loss = SmallDataAggregator()

        for batch in progres_bar:
            if global_step % config.save_every == 0:
                config.debugger.is_debug = True

            losses = train_step(config, batch)
            g_loss.accept_data(losses['g_loss'])
            d_loss.accept_data(losses['d_loss'])
            mel_loss.accept_data(losses['mel_loss'])
            dur_loss.accept_data(losses['dur_loss'])

            progres_bar.set_postfix({
                "G_loss": g_loss.avg(),
                "D_loss": d_loss.avg(),
                "Mel_loss": mel_loss.avg(),
                "Dur_loss": dur_loss.avg(),
            })

            global_step += 1

            if global_step % config.save_every == 0:
                checkpoint = config.model.generator.state_dict()

                torch.save(checkpoint, f"./checkpoints/checkpoint-{global_step}.pth")
                print(f"\n✅ Saved checkpoint at step {global_step}")

            config.debugger.is_debug = False


class SmallDataAggregator:
    def __init__(self):
        self.best = None
        self.worst = None
        self.values = []

    def accept_data(self, data):
        self.best = min(self.best or data, data)
        self.worst = max(self.worst or data, data)

        self.values.append(data)

    def avg(self):
        return f"{sum(self.values) / len(self.values):.4f}"

    def min_max(self):
        return f"{self.best:.4f} - {self.worst:.4f}"
