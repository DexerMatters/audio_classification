import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Custom Dataset for loading and processing audio files with librosa
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
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio file
        waveform, sr = librosa.load(self.file_paths[idx], sr=self.sample_rate)

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

        # Convert to torch tensor
        log_mel_spec = torch.from_numpy(log_mel_spec).float()

        return log_mel_spec, self.labels[idx]
