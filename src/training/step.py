import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.config import TrainingConfig
from src.training.durations import extract_durations_from_alignment


def train_step(config: TrainingConfig, batch):
    audio = batch["audio"].to(config.device)  # [B, max_audio_len]
    mel_spectrogram = batch["mel_spectrogram"].to(config.device)  # [B, n_mels, T_mel]
    input_ids = batch["input_ids"].to(config.device)  # [B, vocab_size]
    text_lengths = batch["text_lengths"].to(config.device)  # [B]
    max_text_len = text_lengths.max().item()
    validate_durations(config, batch)

    debug = config.debugger.debug

    if input_ids.size(-1) < 2:
        input_ids = F.pad(input_ids, (0, 2 - input_ids.size(-1)), value=0)

    input_ids = input_ids[:, :max_text_len + 1]  # [B, max_text_len+1]

    # segment_size = 8192  # Or 16384 depending on your config
    #
    # if audio.size(-1) < segment_size:
    #     # If too short, pad with zeros to reach exactly 8192
    #     pad_len = segment_size - audio.size(-1)
    #     audio = F.pad(audio, (-1, pad_len))
    #     mel_spectrogram = F.pad(mel_spectrogram, (-1, pad_len // config.mel_hop_length))

    model = config.model

    with torch.no_grad():
        text_features = model.generator.emb(input_ids).transpose(1, 2)  # [B, hidden, max_text_len+1]
        text_features = model.generator.encoder(text_features)  # [B, hidden, max_text_len+1]
        text_features = text_features[:, :model.generator.hidden_channels, :]  # [B, hidden, max_text_len+1]

    real_durations = extract_durations_from_alignment(
        text_features, mel_spectrogram[:, :, :math.ceil(audio.size(-1) / config.mel_hop_length)], text_lengths
    )
    debug(f"🔍 Target durations (first 20): {real_durations[0, :20].tolist()}")
    debug(
        f"🔍 Target stats - min: {real_durations.min().item()}, max: {real_durations.max().item()}, mean: {real_durations.float().mean().item():.2f}")
    debug(f"🔍 Target distribution: {(real_durations == 1).float().mean().item() * 100:.1f}% are 1s")

    # Generator pass
    output = model.generator(input_ids=input_ids, real_durations=real_durations)

    y_hat = output["audio_generated"]
    dur_loss = output["dur_loss"]

    # CRITICAL DEBUG: Check y_hat gradients
    debug(f"🔍 y_hat requires_grad: {y_hat.requires_grad}")
    debug(f"🔍 y_hat grad_fn: {y_hat.grad_fn}")

    if y_hat.size(-1) > audio.size(-1):
        y_hat = y_hat[:, :, :audio.size(-1)]
    elif y_hat.size(-1) < audio.size(-1):
        y_hat = F.pad(y_hat, (0, audio.size(-1) - y_hat.size(-1)))

    mel_hat = config.mel_extractor(y_hat).squeeze(1)

    # CRITICAL DEBUG: Check mel_hat gradients
    debug(f"🔍 mel_hat requires_grad: {mel_hat.requires_grad}")
    debug(f"🔍 mel_hat grad_fn: {mel_hat.grad_fn}")
    debug(f"🔍 mel_hat shape: {mel_hat.shape}")
    debug(f"🔍 mel_spectrogram shape: {mel_spectrogram.shape}")

    # extracted mel have to have correct size without padding
    assert mel_spectrogram.size(-1) == mel_hat.size(-1)
    # mel_hat = F.pad(mel_hat[:, :, :mel_spectrogram.size(-1)], (0, mel_spectrogram.size(-1) - mel_hat.size(-1)+1))

    mel_loss = nn.L1Loss()(mel_hat, mel_spectrogram)
    total_g_loss = mel_loss * 5 + dur_loss * 20.0

    # Discriminator update
    config.optimizer_d.zero_grad()
    y_hat_detached = y_hat.detach()  # [B, 1, audio_len]

    d_real, _ = model.discriminator(audio.unsqueeze(1) if audio.dim() == 2 else audio)
    d_fake, _ = model.discriminator(y_hat_detached)

    ## Hinge loss
    loss_d_real = sum(torch.mean((d - 1.0) ** 2) for d in d_real) / len(d_real)
    loss_d_fake = sum(torch.mean((d + 1.0) ** 2) for d in d_fake) / len(d_fake)
    total_d_loss = (loss_d_real + loss_d_fake) / 2

    # Backprop
    total_d_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
    config.optimizer_d.step()
    # config.scheduler_d.step()

    # Generator update
    config.optimizer_g.zero_grad()

    _, f_real = model.discriminator(audio.unsqueeze(1) if audio.dim() == 2 else audio,
                                    return_features=True)
    d_fake, f_fake = model.discriminator(y_hat,
                                         return_features=True)

    # CRITICAL DEBUG: Check discriminator outputs
    debug(f"🔍 d_fake values: {[d.mean().item() for d in d_fake]}")
    debug(f"🔍 d_real values: {[d.mean().item() for d in d_real]}")
    ## Adversarial loss. Need to have higher values on fake values
    ## Discriminator tries to push d_fake to -1, so adv_loss should be 1
    # so that generator learns how to fool discriminator
    adv_loss = sum(torch.mean((d - 1.0) ** 2) for d in d_fake) / len(d_fake)
    debug(f"🔍 adv_loss: {adv_loss.item()}")

    # Feature-matching loss
    fm_losses = []
    for drs, dfs in zip(f_real, f_fake):
        fm_loss = 0
        for dr, df in zip(drs, dfs):
            fm_loss += nn.L1Loss()(dr, df)
        fm_loss = fm_loss / len(drs)
        fm_losses.append(fm_loss)

    full_fm_loss = (sum(fm_losses) / len(fm_losses)) if len(fm_losses) > 0 else 0
    debug(f"🔍 full_fm_loss: {full_fm_loss.item()}")

    final_g_loss = total_g_loss + adv_loss*10 + full_fm_loss * 5
    debug(f"🔍 final_g_loss: {final_g_loss.item()}")
    debug(f"🔍 final_g_loss grad_fn: {final_g_loss.grad_fn}")

    # Backprop
    final_g_loss.backward()

    # CRITICAL DEBUG: Check if gradients reached the generator
    grad_norms = []
    for name, param in model.generator.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    if grad_norms:
        debug(
            f"🔍 Generator grad norms - min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}, mean: {sum(grad_norms) / len(grad_norms):.6f}")
    else:
        debug(f"🔍 ❌ NO GRADIENTS in generator!")

    # torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
    config.optimizer_g.step()
    # config.scheduler_g.step()

    return {
        "g_loss": final_g_loss.item(),
        "d_loss": total_d_loss.item(),
        "mel_loss": mel_loss.item(),
        "dur_loss": dur_loss.item(),
    }

def validate_durations(config, batch):
    audio = batch["audio"].to(config.device)
    mel_spectrogram = batch["mel_spectrogram"].to(config.device)
    input_ids = batch["input_ids"].to(config.device)
    text_lengths = batch["text_lengths"].to(config.device)

    model = config.model
    with torch.no_grad():
        text_features = model.generator.emb(input_ids).transpose(1, 2)
        text_features = model.generator.encoder(text_features)
        real_durations = extract_durations_from_alignment(
            text_features, mel_spectrogram, text_lengths
        )

    # Calculate expected audio length
    expected_audio_len = torch.max(torch.sum(real_durations, dim=1)) * config.mel_hop_length

    # Check if audio length matches expected
    if audio.size(-1) != expected_audio_len:
        config.debugger.debug(f"⚠️ DURATION MISMATCH: Actual audio len={audio.size(-1)}, Expected={expected_audio_len}")
        config.debugger.debug(f"  Durations: {real_durations[0, :20].tolist()}")
        config.debugger.debug(f"  Mel len: {mel_spectrogram.size(-1)}")