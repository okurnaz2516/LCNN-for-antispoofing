# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 23:16:46 2024

@author: chanilci
"""

import torch
import torch.nn.functional as F
# Function to save output probabilities of the validation dataset to a file
def save_output_probabilities(model, val_loader, output_file='output_probabilities.txt', device='cpu'):
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs, embeddings = model(spectrograms)
            outputs = F.sigmoid(outputs)
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    with open(output_file, 'w') as f:
        for output, label in zip(all_outputs, all_labels):
            probabilities = ' '.join(map(str, output))
            #f.write(f'Label: {label}, Probabilities: {probabilities}\n')
            f.write(f'{label} {probabilities}\n')
            
    return all_outputs, all_labels