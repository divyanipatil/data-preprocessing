import torchaudio
import torch
import random


class AudioAugmenter:
    def __init__(self):
        self.sample_rate = 16000

    def time_shift(self, waveform, shift_factor=0.2):
        """Shift audio in time"""
        length = waveform.shape[1]
        shift_amount = int(length * shift_factor)
        shifted = torch.roll(waveform, shifts=shift_amount, dims=1)
        return shifted

    def pitch_shift(self, waveform, pitch_factor=2):
        """Shift the pitch of the audio"""
        effects = [
            ["pitch", str(pitch_factor * 100)],  # multiply by 100 as pitch is in cents
            ["rate", str(self.sample_rate)]
        ]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return augmented
