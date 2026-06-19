from debugger import Debugger
from src.model import SimpleVITS


class TrainingConfig:
    def __init__(self, tokenizer, debugger: Debugger, vocab_size, model: SimpleVITS, optimizer_g, optimizer_d, mel_extractor, device, min_audio_length,
                 mel_hop_length, num_epochs, save_every, scheduler_g = None, scheduler_d=None):
        self.model = model
        self.debugger = debugger
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.mel_hop_length = mel_hop_length
        self.min_audio_length = min_audio_length
        self.device = device
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.mel_extractor = mel_extractor
