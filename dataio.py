
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
        audio_path = self.audiopath + self.data[idx, 0]
        key = self.data[idx,1]
        if key=='genuine':
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

        # Normalize the spectrogram
        # mean = torch.mean(spectrogram,2)
        # std = torch.std(spectrogram,2)
        # mean = mean.repeat(spectrogram.size(2),1)
        # mean = torch.transpose(mean,0,1)
        # std = std.repeat(spectrogram.size(2),1)
        # std = torch.transpose(std,0,1)
        
        # #mean = spectrogram.mean()
        # #std = spectrogram.std()
        # spectrogram = (spectrogram - mean) / (std + 1e-6)
        
        return spectrogram, label

    def _fix_num_frames(self, spectrogram, num_frames):
        if spectrogram.shape[-1] < num_frames:
            pad_amount = num_frames - spectrogram.shape[-1]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_amount))
        else:
            spectrogram = spectrogram[:, :, :num_frames]
        return spectrogram
