

import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

labels = []
scores = []
with open('./lcnn_eval_scores.txt','r') as f:
    lines = f.readlines()

for line in lines:
    data = line.rsplit()
    key = int(data[0])
    s1 = float(data[1])
    # s2 = float(data[2])
    labels.append(key)
    # temp = np.log(s2+np.finfo(float).eps) - np.log(s1+np.finfo(float).eps)
    scores.append(s1)

scores = np.array(scores)
labels = np.array(labels)

genuine_scores = scores[labels==1]
spoof_scores = scores[labels==0]

import seaborn as sns
sns.distplot(genuine_scores, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1.5}, 
                  label = 'bonafide')
sns.distplot(spoof_scores, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1.5}, 
                  label = 'spoof')

fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
print(eer*100)
