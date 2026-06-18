import numpy as np
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class VitsAudioDataset(Dataset):
    def __init__(self, data: list[tuple[str, str]], tokenizer, mel_extractor, device, min_audio_length,
                 sample_rate=16000):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.mel_extractor = mel_extractor
        self.min_audio_length = min_audio_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        audio_path, text = self.data[id]
        sr, waveform_np = wavfile.read(audio_path)

        # Convert numpy array to PyTorch tensor
        waveform = torch.from_numpy(waveform_np).float().to(self.device)

        # If the audio is stereo (shape: [samples, 2]), convert it to mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)

        # Normalize 16-bit PCM audio to [-1.0, 1.0]
        if waveform_np.dtype == np.int16:
            waveform = waveform / 32768.0

        # SciPy shape is [Time, Channels] for stereo, [Time] for mono.
        if waveform.dim() > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1)  # Average channels

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform_np.shape[-1] < self.min_audio_length:
            pad_amount = self.min_audio_length - waveform_np.shape[-1]
            waveform = F.pad(waveform, (0, pad_amount))

        waveform = waveform.unsqueeze(0)
        mel_spectrogram = self.mel_extractor(waveform).squeeze(0)

        encoding = self.tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

        input_ids_tensor = encoding["input_ids"].squeeze(0)

        if input_ids_tensor.dim() == 0 or len(input_ids_tensor) < 2:
            input_ids_tensor = torch.tensor([self.tokenizer.pad_token_id] * 2)

        return {
            "audio": waveform.squeeze(0),
            "mel_spectrogram": mel_spectrogram.to(self.device),
            "input_ids": input_ids_tensor.detach().clone().to(self.device),
            "attention_mask": encoding["attention_mask"].squeeze(0).detach().clone().to(self.device),
            "audio_lengths": waveform.shape[-1],
            "text_lengths": (input_ids_tensor != self.tokenizer.pad_token_id).sum().item()
        }


def vits_collate_fn(batch):
    # 1. Separate the batch into lists
    audios = [item["audio"] for item in batch]
    mel_specs = [item["mel_spectrogram"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]

    audio_lens = torch.tensor([item["audio_lengths"] for item in batch], dtype=torch.long)
    text_lens = torch.tensor([item["text_lengths"] for item in batch], dtype=torch.long)

    # 2. Pad Audio (pad with 0.0)
    # Shape: [batch_size, max_audio_len]
    audio_padded = rnn_utils.pad_sequence(audios, batch_first=True, padding_value=0.0)

    # 3. Pad Mel Spectrograms (pad with 0.0)
    # Shape: [batch_size, n_mels, max_mel_len]
    # Note: pad_sequence expects [seq_len, ...], so we transpose, pad, and transpose back
    mel_specs_transposed = [m.transpose(0, 1) for m in mel_specs]
    mel_padded_transposed = rnn_utils.pad_sequence(mel_specs_transposed, batch_first=True, padding_value=0.0)
    mel_padded = mel_padded_transposed.transpose(1, 2)  # Back to [batch, n_mels, time]

    # 4. Pad Input IDs (pad with tokenizer's pad_token_id, usually 0)
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {
        "audio": audio_padded,
        "mel_spectrogram": mel_padded,
        "input_ids": input_ids_padded,

        "audio_lengths": audio_lens,
        "text_lengths": text_lens
    }


def create_data_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=vits_collate_fn,
        num_workers=0,
        pin_memory=False
    )
