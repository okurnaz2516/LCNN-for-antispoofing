# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 23:03:24 2024

@author: chanilci
"""
import torchaudio 
import torch

class SpectrogramTransform:
    def __init__(self, n_fft=512, win_length=None, hop_length=None):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def __call__(self, waveform):
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            normalized=True
        )(waveform)
        spectrogram = torch.log(spectrogram+1e-17)
        # m = torch.mean(spectrogram, 1)
        # m = m.repeat(spectrogram.size(1),1)
        # m = torch.transpose(m, 0, 1)
        # s = torch.std(spectrogram,1)
        # s = s.repeat(spectrogram.size(1),1)
        # s = torch.transpose(s, 0, 1)
        # spectrogram = (spectrogram - m)/s
        #log_spectrogram = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)(spectrogram)
        return spectrogram