import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.config import TrainingConfig
from src.training.durations import extract_durations_from_alignment


def train_step(config: TrainingConfig, batch):
    audio = batch["audio"].to(config.device) # [B, max_audio_len]
    mel_spectrogram = batch["mel_spectrogram"].to(config.device) # [B, n_mels, T_mel]
    input_ids = batch["input_ids"].to(config.device) # [B, vocab_size]
    text_lengths = batch["text_lengths"].to(config.device) # [B]
    max_text_len = text_lengths.max().item()

    if input_ids.size(-1) < 2:
        input_ids = F.pad(input_ids, (0, 2 - input_ids.size(-1)), value=0)

    input_ids = input_ids[:, :max_text_len + 1] # [B, max_text_len+1]

    # segment_size = 8192  # Or 16384 depending on your config
    #
    # if audio.size(-1) < segment_size:
    #     # If too short, pad with zeros to reach exactly 8192
    #     pad_len = segment_size - audio.size(-1)
    #     audio = F.pad(audio, (-1, pad_len))
    #     mel_spectrogram = F.pad(mel_spectrogram, (-1, pad_len // config.mel_hop_length))

    model = config.model

    with torch.no_grad():
        text_features = model.generator.emb(input_ids).transpose(1,2) # [B, hidden, max_text_len+1]
        text_features = model.generator.encoder(text_features) # [B, hidden, max_text_len+1]
        text_features = text_features[:, :model.generator.hidden_channels, :] # [B, hidden, max_text_len+1]

    real_durations = extract_durations_from_alignment(
        text_features, mel_spectrogram, text_lengths
    )

    # Generator pass
    output = model.generator(input_ids=input_ids, real_durations=real_durations)

    y_hat = output["audio_generated"]
    dur_loss = output["dur_loss"]

    if y_hat.size(-1) > audio.size(-1):
        y_hat = y_hat[:, :, :audio.size(-1)]
    elif y_hat.size(-1) < audio.size(-1):
        y_hat = F.pad(y_hat, (-1, audio.size(-1) - y_hat.size(-1)+1))

    mel_hat = config.mel_extractor(y_hat).squeeze(1)
    # extracted mel have to have correct size without padding
    mel_hat = F.pad(mel_hat[:, :, :mel_spectrogram.size(-1)], (-1, mel_spectrogram.size(-1) - mel_hat.size(-1)+1))

    mel_loss = nn.L1Loss()(mel_hat, mel_spectrogram)
    total_g_loss = mel_loss * 10 + dur_loss * 2.0

    # Discriminator update
    config.optimizer_d.zero_grad()
    y_hat_detached = y_hat.detach() # [B, 1, audio_len]

    d_real, _ = model.discriminator(audio.unsqueeze(1) if audio.dim() == 2 else audio)
    d_fake, _ = model.discriminator(y_hat_detached)

    ## Hinge loss
    loss_d_real = sum(torch.mean(torch.relu(0.9 - d)) for d in d_real) / len(d_real)
    loss_d_fake = sum(torch.mean(torch.relu(1.0 + d)) for d in d_fake) / len(d_fake)
    total_d_loss = loss_d_real + loss_d_fake

    # Backprop
    total_d_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
    config.optimizer_d.step()
    config.scheduler_d.step()

    # Generator update
    config.optimizer_g.zero_grad()

    _, f_real = model.discriminator(audio.unsqueeze(1) if audio.dim() == 2 else audio,
                                    return_features=True)
    d_fake, f_fake = model.discriminator(y_hat,
                                         return_features=True)

    ## Adversarial loss. Need to have higher values on fake values
    ## Discriminator tries to push d_fake to -1, so adv_loss should be 1
    # so that generator learns how to fool discriminator
    adv_loss = -sum(torch.mean(d) for d in d_fake) / len(d_fake)

    # Feature-matching loss
    fm_losses = []
    for drs, dfs in zip(f_real, f_fake):
        fm_loss = 0
        for dr, df in zip(drs, dfs):
            fm_loss += nn.L1Loss()(dr, df)
        fm_loss = fm_loss / len(drs)
        fm_losses.append(fm_loss)

    full_fm_loss = (sum(fm_losses) / len(fm_losses)) if len(fm_losses) > 0 else 0

    final_g_loss = total_g_loss + adv_loss + full_fm_loss * 2

    if final_g_loss > 400 or torch.isnan(final_g_loss):
        config.optimizer_g.zero_grad()
        config.optimizer_d.zero_grad()
        return {
            "g_loss": 0.0,
            "d_loss": 0.0,
            "mel_loss": 0.0,
            "dur_loss": 0.0,
            "skipped": True
        }

    # Backprop
    final_g_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
    config.optimizer_g.step()
    config.scheduler_g.step()

    if torch.isnan(final_g_loss) or torch.isnan(total_d_loss):
        print(
            f"⚠️ NaN detected!: final_g_loss: {final_g_loss}, total_d_loss: {total_d_loss} Skipping batch to save model weights.")
        config.optimizer_g.zero_grad()
        config.optimizer_d.zero_grad()
        return {"g_loss": 0.0, "d_loss": 0.0, "mel_loss": 0.0, "dur_loss": 0.0}

    return {
        "g_loss": final_g_loss.item(),
        "d_loss": total_d_loss.item(),
        "mel_loss": mel_loss.item(),
        "dur_loss": dur_loss.item(),
    }
