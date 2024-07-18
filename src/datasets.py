import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from sklearn.preprocessing import StandardScaler

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).to(torch.float32)
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt")).long()
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        # データ前処理
        self.X = self.preprocess(self.X)

    def preprocess(self, X):
        # リサンプリング (サンプルレートを200Hzに変換)
        X = self.resample(X, 200).to(torch.float32)

        # フィルタリング (例: 0.1-40Hzのバンドパスフィルタ)
        X = self.bandpass_filter(X, 0.1, 40, 200).float()

        # スケーリング (標準化)
        scaler = StandardScaler()
        X = scaler.fit_transform(X.numpy().reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
        X = torch.tensor(X, dtype=torch.float32)

        # ベースライン補正
        X = self.baseline_correction(X).to(torch.float32)

        return X

    def resample(self, X, target_fs):
        # サンプルコード: scipy.signal.resample を使用
        from scipy.signal import resample
        num_samples = int(X.shape[-1] * target_fs / 1000)  # Assuming original fs is 1000Hz
        return torch.tensor(resample(X, num_samples, axis=-1), dtype=torch.float32)

    def bandpass_filter(self, X, lowcut, highcut, fs):
        from scipy.signal import butter, filtfilt
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(1, [low, high], btype='band')
        X_filtered = filtfilt(b, a, X.numpy(), axis=-1)
        return torch.tensor(X_filtered.copy(), dtype=torch.float32)  # Add .copy() to avoid negative strides

    def baseline_correction(self, X):
        baseline = X[..., :100].mean(axis=-1, keepdims=True)
        return X - baseline

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
