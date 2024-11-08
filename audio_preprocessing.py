import torchaudio
import torch
import torch.nn.functional as F
import soundfile as sf
import base64
import io
import numpy as np


class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000  # Standard sample rate
        # Set the backend to soundfile
        torchaudio.set_audio_backend("soundfile")

    def load_audio(self, audio_bytes):
        """Load audio from bytes"""
        try:
            # Create a BytesIO object from the audio bytes
            buffer = io.BytesIO(audio_bytes)

            # Read audio data using soundfile
            data, samplerate = sf.read(buffer)

            # Convert to tensor and reshape if needed
            if len(data.shape) == 1:
                waveform = torch.FloatTensor(data).unsqueeze(0)
            else:
                waveform = torch.FloatTensor(data.T)

            # Resample if necessary
            if samplerate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(samplerate, self.sample_rate)
                waveform = resampler(waveform)

            return waveform, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Error loading audio: {str(e)}")

    def add_noise(self, waveform, noise_level=0.005):
        """Add Gaussian noise to the audio"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def change_speed(self, waveform, speed_factor=1.6):
        """Change the speed of the audio"""
        try:
            effects = [
                ["speed", str(speed_factor)],
                ["rate", str(self.sample_rate)]
            ]
            augmented, new_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects
            )
            return augmented
        except Exception as e:
            print(f"Warning: Speed change failed, returning original. Error: {str(e)}")
            return waveform

    def apply_low_pass_filter(self, waveform, cutoff_freq=4000):
        """Apply a low-pass filter"""
        try:
            # Ensure waveform is in proper shape (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # Design basic low-pass filter
            freq_bins = torch.fft.fftfreq(waveform.shape[-1])
            mask = torch.abs(freq_bins) < cutoff_freq / self.sample_rate
            mask = mask.float()

            # Apply filter in frequency domain
            fft = torch.fft.fft(waveform)
            filtered_fft = fft * mask
            filtered = torch.fft.ifft(filtered_fft).real

            return filtered
        except Exception as e:
            print(f"Warning: Low-pass filter failed, returning original. Error: {str(e)}")
            return waveform

    def to_base64(self, waveform, sample_rate):
        """Convert audio tensor to base64 string"""
        try:
            # Convert to numpy array
            audio_numpy = waveform.numpy()

            # Create a BytesIO buffer
            buffer = io.BytesIO()

            # Save as WAV using soundfile
            if audio_numpy.ndim == 1:
                audio_numpy = audio_numpy.reshape(1, -1)
            sf.write(buffer, audio_numpy.T, sample_rate, format='WAV')

            # Get base64 string
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode()

            return audio_b64

        except Exception as e:
            raise RuntimeError(f"Error converting to base64: {str(e)}")
