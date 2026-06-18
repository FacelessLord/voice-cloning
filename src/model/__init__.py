import torch.nn as nn
from src.model.generator import SimplifiedVITSGenerator
from src.model.discriminator import MultiperiodDiscriminator


class SimpleVITS(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.generator = SimplifiedVITSGenerator(vocab_size, hidden_channels=192)
        self.discriminator = MultiperiodDiscriminator()
