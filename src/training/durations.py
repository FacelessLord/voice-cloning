import torch
import torch.nn as nn
import torch.nn.functional as F

# Cache the projection layer to avoid recreating it every call
_mel_projection_cache = {}

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

    # Cache the projection layer
    cache_key = (n_mels, C, text_features.device)
    if cache_key not in _mel_projection_cache:
        _mel_projection_cache[cache_key] = nn.Linear(n_mels, C).to(text_features.device)

    mel_to_hidden = _mel_projection_cache[cache_key]
    mel_proj = mel_to_hidden(mel_spectrogram.transpose(1, 2))  # [B, T_mel, C]
    mel_proj = F.normalize(mel_proj, dim=2)
    text_norm = F.normalize(text_features, dim=1)

    # Compute similarity matrix: [B, T_text, T_mel]
    similarity = torch.bmm(text_norm.transpose(1, 2), mel_proj.transpose(1, 2))

    # Extract durations: for each text token, count how many mel frames it "owns"
    # Use argmax to find the best matching mel frame for each text token
    # Then count consecutive assignments
    durations = torch.zeros(B, T_text, device=text_features.device, dtype=torch.long)  # [B, T_text]

    for b in range(B):
        text_len = text_lengths[b].item()

        # SAFETY: Handle empty text
        if text_len == 0:
            continue

        attn = similarity[b, :text_len, :]
        # Compute total similarity score for each token
        # Shape: [T_text]
        token_scores = similarity[b, :text_len, :].sum(dim=1)

        # Normalize scores to get proportions
        # Add small epsilon to avoid division by zero
        total_score = token_scores.sum() + 1e-8
        proportions = token_scores / total_score

        # Allocate mel frames proportionally
        # Start with base allocation
        base_durations = (proportions * T_mel).round().long().clamp(min=2)

        # Adjust to match exact T_mel
        total_dur = base_durations.sum()
        diff = T_mel - total_dur

        if diff > 0:
            # Need to add more frames - distribute to tokens with highest scores
            sorted_indices = torch.argsort(token_scores, descending=True)
            for i in range(diff):
                idx = sorted_indices[i % text_len].item()
                base_durations[idx] += 1
        elif diff < 0:
            # Need to remove frames - remove from tokens with lowest scores (but keep min=2)
            sorted_indices = torch.argsort(token_scores)
            removed = 0
            for idx in sorted_indices:
                idx = idx.item()
                can_remove = base_durations[idx] - 2
                if can_remove > 0 and removed < abs(diff):
                    remove_amount = min(can_remove, abs(diff) - removed)
                    base_durations[idx] -= remove_amount
                    removed += remove_amount
                if removed >= abs(diff):
                    break

        durations[b, :text_len] = base_durations

    return durations


def monotonic_alignment_search(log_p: torch.Tensor) -> torch.Tensor:
    """
    Monotonic Alignment Search using dynamic programming.

    Args:
        log_p: [T_text, T_mel] - negative log probability (similarity) matrix
               Lower values = better alignment

    Returns:
        durations: [T_text] - duration for each text token
    """
    T_text, T_mel = log_p.shape

    # Dynamic programming table
    # Q[t, m] = minimum cost to align text[0:t] to mel[0:m]
    Q = torch.full((T_text, T_mel), float('inf'), device=log_p.device)

    # Base case: first text token
    Q[0, :] = torch.cumsum(log_p[0, :], dim=0)

    # Fill the DP table
    for t in range(1, T_text):
        # Option 1: extend previous text token's alignment
        # Option 2: start new text token at this mel frame
        Q[t, 1:] = log_p[t, 1:] + torch.minimum(Q[t - 1, :-1], Q[t, :- 1])

    # Backtrack to find the optimal path
    path = torch.zeros(T_text, dtype=torch.long, device=log_p.device)
    m = T_mel - 1

    for t in range(T_text - 1, -1, -1):
        path[t] = m

        if t == 0:
            break

        for m_prev in range(m - 1, t - 1, -1):
            # Check if this is where text[t] started
            # It started here if Q[t-1, m_prev-1] was used (not Q[t, m_prev])
            if m_prev > 0 and Q[t - 1, m_prev - 1] < Q[t, m_prev]:
                m = m_prev - 1
                break
        else:
            # No transition found, assign minimum
            m = t - 1

    durations = torch.zeros(T_text, dtype=torch.long, device=log_p.device)

    for t in range(T_text):
        if t == 0:
            start = 0
        else:
            start = path[t - 1] + 1
        end = path[t]
        durations[t] = end - start + 1

    return durations
