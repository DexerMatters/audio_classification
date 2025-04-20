import librosa
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        file_paths,
        labels,
        sample_rate=16000,
        duration=10,
        n_mels=128,
        n_fft=400,
        hop_length=160,
        augment=False,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio file
        waveform, sr = librosa.load(self.file_paths[idx], sr=self.sample_rate)

        # Apply data augmentation if enabled
        if self.augment:
            waveform = self._apply_augmentation(waveform)

        # Pad or truncate to target length
        target_length = self.sample_rate * self.duration
        if len(waveform) < target_length:
            padding = target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), "constant")
        else:
            waveform = waveform[:target_length]

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (seq_len, n_mels)
        log_mel_spec = log_mel_spec.T

        # Add frequency masking (helps with overfitting)
        if self.augment:
            log_mel_spec = self._apply_freq_masking(log_mel_spec)

        # Convert to torch tensor
        log_mel_spec = torch.from_numpy(log_mel_spec).float()

        return log_mel_spec, self.labels[idx]

    def _apply_augmentation(self, waveform):
        """Apply time-domain augmentations to the audio waveform"""
        # Random time shift
        if random.random() < 0.5:
            shift_amount = int(random.random() * self.sample_rate)
            waveform = np.roll(waveform, shift_amount)

        # Random pitch shift (between -2 and 2 semitones)
        if random.random() < 0.5:
            pitch_shift = random.uniform(-2.0, 2.0)
            waveform = librosa.effects.pitch_shift(
                waveform, sr=self.sample_rate, n_steps=pitch_shift
            )

        # Random volume change
        if random.random() < 0.5:
            volume_factor = random.uniform(0.75, 1.25)
            waveform = waveform * volume_factor

        return waveform

    def _apply_freq_masking(self, mel_spec):
        """Apply frequency masking to the mel spectrogram"""
        freq_width = int(random.uniform(1, self.n_mels * 0.15))
        freq_start = random.randint(0, self.n_mels - freq_width - 1)

        mel_spec_copy = mel_spec.copy()
        mel_spec_copy[:, freq_start : freq_start + freq_width] = mel_spec_copy.mean()

        return mel_spec_copy
