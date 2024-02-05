import os
import pandas as pd

import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
import pytorch_lightning as pl

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, target_sample_rate, num_samples, transformation):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal
        # signal -> (num_channels,samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1,50000) -> (1,22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1,1,1] -> [1,1,1,0,0]
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2,16000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

class UrbanSoundDataModule(pl.LightningDataModule):
    def __init__(self, annotations_file, audio_dir, target_sample_rate, num_samples, pin_memory=False, num_workers=1, batch_size=32):
        super().__init__()
        self.annotations_file = annotations_file
        self.audio_dir = audio_dir
        self.sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.setup()
    
    # Called on one GPU
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Split the dataset into train and validation sets
        full_dataset = UrbanSoundDataset(self.annotations_file, self.audio_dir, self.sample_rate, self.num_samples, self.transformation)
        test_split = int(0.05 * len(full_dataset))
        val_split = int(0.05 * len(full_dataset))
        train_split = len(full_dataset) - (test_split + val_split)
        self.train, self.val, self.test = random_split(full_dataset, [train_split, val_split, test_split])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers)
    
    def get_shape(self):
        for batch in self.train_dataloader():
            X, y = batch
            break
        return X.shape, y.shape