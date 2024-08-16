
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, protocol_file, AudioPath, transform=None, fixed_num_frames=None):
        self.data = np.loadtxt(protocol_file,dtype='str')
        self.audiopath = AudioPath
        self.transform = transform
        self.fixed_num_frames = fixed_num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.audiopath + self.data[idx, 1] + '.flac'
        key = self.data[idx,4]
        if key=='bonafide':
            label = 1
        else:
            label = 0

        #label = self.data.iloc[idx, 1]
        num_frames = self.fixed_num_frames

        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            spectrogram = self.transform(waveform)
            spectrogram = self._fix_num_frames(spectrogram, num_frames)
        else:
            spectrogram = waveform

        return spectrogram, label

    def _fix_num_frames(self, spectrogram, num_frames):
        if spectrogram.shape[-1] < num_frames:
              while spectrogram.shape[-1] < num_frames:
                  remaining_pad = num_frames - spectrogram.shape[-1]
                  pad_part = spectrogram[:, :, :min(spectrogram.shape[-1], remaining_pad)]
                  spectrogram = torch.cat([spectrogram, pad_part], dim=-1)
        else:
            spectrogram = spectrogram[:, :, :num_frames]
        return spectrogram
