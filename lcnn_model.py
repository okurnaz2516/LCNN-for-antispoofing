# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 01:28:51 2024

@author: chanilci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the MaxFeatureMap class
class MaxFeatureMap(nn.Module):
    def __init__(self):
        super(MaxFeatureMap, self).__init__()

    def forward(self, x):
        n_channels = x.size(1)
        assert n_channels % 2 == 0, "The number of input channels must be even."
        return torch.max(x[:, :n_channels//2], x[:, n_channels//2:])
    
# Define the Max-Feature-Map activation function
# class MaxFeatureMap(nn.Module):
#     def forward(self, x):
#         out = torch.max(x[:, ::2, :, :], x[:, 1::2, :, :])
#         return out

# Define the LCNN model
class LCNN(nn.Module):
    def __init__(self, input_dim=(257, 750), num_classes=2):
        super(LCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.mfm1 = MaxFeatureMap()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2a.weight)
        nn.init.zeros_(self.conv2a.bias)
        self.mfm2a = MaxFeatureMap()
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.conv2b = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2b.weight)
        nn.init.zeros_(self.conv2b.bias)
        self.mfm2b = MaxFeatureMap()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(48, affine=False)
        
        self.conv3a = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv3a.weight)
        nn.init.zeros_(self.conv3a.bias)
        self.mfm3a = MaxFeatureMap()
        self.bn3 = nn.BatchNorm2d(48, affine=False)
        self.conv3b = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv3b.weight)
        nn.init.zeros_(self.conv3b.bias)
        self.mfm3b = MaxFeatureMap()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv4a.weight)
        nn.init.zeros_(self.conv4a.bias)
        self.mfm4a = MaxFeatureMap()
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv4b.weight)
        nn.init.zeros_(self.conv4b.bias)
        self.mfm4b = MaxFeatureMap()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(32, affine=False)
        
        self.conv5a = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv5a.weight)
        nn.init.zeros_(self.conv5a.bias)
        self.mfm5a = MaxFeatureMap()
        self.bn6 = nn.BatchNorm2d(32,affine=False)
        self.conv5b = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv5b.weight)
        nn.init.zeros_(self.conv5b.bias)
        self.mfm5b = MaxFeatureMap()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output size of the final convolutional layer
        conv_output_size = self._get_conv_output(input_dim)
        self.dropout1 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(conv_output_size, 160)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.mfm_fc1 = MaxFeatureMap()
        self.bnfc1 = nn.BatchNorm1d(80, affine=False)
        self.fc2 = nn.Linear(80, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *shape)
            dummy_output = self._forward_conv(dummy_input)
            return int(torch.prod(torch.tensor(dummy_output.size()[1:])))
        

    def _forward_conv(self, x):
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.bn2(self.pool2(self.mfm2b(self.conv2b(self.bn1(self.mfm2a(self.conv2a(x)))))))
        x = self.pool3(self.mfm3b(self.conv3b(self.bn3(self.mfm3a(self.conv3a(x))))))
        x = self.bn5(self.pool4(self.mfm4b(self.conv4b(self.bn4(self.mfm4a(self.conv4a(x)))))))
        x = self.pool5(self.mfm5b(self.conv5b(self.bn6(self.mfm5a(self.conv5a(x))))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc1(x)
        x = self.mfm_fc1(embedding)
        x = self.dropout1(x)
        x = self.bnfc1(x)
        x = self.fc2(x)
        return x, embedding   #F.softmax(x, -1)
