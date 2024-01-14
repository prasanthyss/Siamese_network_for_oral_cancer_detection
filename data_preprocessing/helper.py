from __future__ import print_function, division
import os
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import copy

device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")

class finalLayer(nn.Module):
	def __init__(self, featureSize):
		super(finalLayer, self).__init__()
		self.layer = nn.Linear(featureSize, 1)
	def forward(self, x1, x2):
		dstnc = torch.abs(x1-x2)
		out = self.layer(dstnc)
		return out.sigmoid()

FL = finalLayer(4096)
for params in FL.parameters():
	print('ho')
FL.to(device)
a = torch.randn(4, 4096)
b = torch.randn(4, 4096).to(device)
c = FL(a, b)
print(c)

