# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 23:06:14 2024

@author: chanilci
"""

import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
import torch.nn.functional as F
from scipy.optimize import brentq
# Function to calculate EER
def calculate_eer(outputs, labels):
    outputs = F.sigmoid(outputs)
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    scores = outputs
    #scores = np.log(outputs[:,1]+np.finfo(float).eps) - np.log(outputs[:,0]+np.finfo(float).eps)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    return eer*100