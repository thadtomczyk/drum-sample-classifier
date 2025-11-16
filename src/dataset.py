import os
import glob
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["kick", "snare", "clap", "hat", "perc"]

class DrumSampleDataset(Dataset):
    def __init__(self, root_dir, split="train", split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.sample_rate = 22050
        self.n_mels = 64
        self.max_duration = 0.5  # seconds

        self.files = []
        self.labels = []

        for idx, cls in enumerate(CLASS_NAMES):
            pattern = os.path.join(root_dir, cls, "*.wav")
            wavs = sorted(glob.glob(pattern))
            for w in wavs:
                self.files.append(w)
                self.labels.append(idx)

        paired = list(zip(self.files, self.labels))
        random.Random(seed).shuffle(paired)
        if paired:
            self.files, self.labels = zip(*paired)
        else:
            self.files, self.labels = [], []

        n_train = int(len(self.files) * split_ratio)
        if split == "train":
            self.files = self.files[:n_train]
            self.labels = self.labels[:n_train]
        else:
            self.files = self.files[n_train:]
            self.labels = self.labels[n_train:]

    def __len__(self):
        return len(self.files)

    def _load_mel(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        max_len = int(self.max_duration * self.sample_rate)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            pad = max_len - len(y)
            y = np.pad(y, (0, pad))

        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db + 80.0) / 80.0  # rough normalization to [0, 1]
        return S_db.astype(np.float32)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        mel = self._load_mel(path)
        mel = np.expand_dims(mel, axis=0)  # [1, n_mels, time]
        return torch.from_numpy(mel), torch.tensor(label, dtype=torch.long)

