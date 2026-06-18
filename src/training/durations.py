import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_durations_from_alignment(text_features, mel_spectrogram, text_lengths):
    """
    Extract ground truth durations by computing attention between text and mel.

    Args:
        text_features: [B, hidden, T_text] - encoded text embeddings
        mel_spectrogram: [B, n_mels, T_mel] - target mel spectrogram
        text_lengths: [B] - actual (unpadded) text lengths

    Returns:
        durations: [B, T_text] - ground truth durations (how many mel frames per token)
    """
    B, C, T_text = text_features.shape
    _, n_mels, T_mel = mel_spectrogram.shape

    text_norm = F.normalize(text_features, dim=1)

    mel_to_hidden = nn.Linear(n_mels, C).to(text_features.device)
    mel_proj = mel_to_hidden(mel_spectrogram.transpose(1, 2)) # [B, T_mel, C]
    mel_proj = F.normalize(mel_proj, dim=2)

    # Compute similarity matrix: [B, T_text, T_mel]
    similarity = torch.bmm(text_norm.transpose(1, 2), mel_proj.transpose(1, 2))

    # Apply softmax over mel dimension (each text token attends to mel frames)
    attention = F.softmax(similarity, dim=2) # [B, T_text, T_mel]

    # Extract durations: for each text token, count how many mel frames it "owns"
    # Use argmax to find the best matching mel frame for each text token
    # Then count consecutive assignments
    durations = torch.zeros(B, T_text, device=text_features.device, dtype=torch.long) # [B, T_text]

    for b in range(B):
        text_len = text_lengths[b].item()
        if text_len == 0:
            continue

        # Get attention weights for this batch item
        attn = attention[b, :text_len, :] # [text_len, T_mel]

        # For each mel frame, find which text token it attends to most
        mel_to_text = torch.argmax(attn, dim=0) # [T_mel]

        # Count how many mel frames are assigned to each text token
        for t in range(text_len):
            durations[b, t] = (mel_to_text == t).sum()

    durations = durations.clamp(min=1)

    for b in range(B):
        text_len = text_lengths[b].item()
        if text_len == 0:
            continue

        #  How long did this text sound
        total_dur = durations[b, :text_len].sum()
        diff = T_mel - total_dur

        if diff > 0:
            durations[b, text_len - 1] += diff
        elif diff < 0:
            durations[b, text_len - 1] = max(1, durations[b, text_len - 1] + diff)

    return durations
