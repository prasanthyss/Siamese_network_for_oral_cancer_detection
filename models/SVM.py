import os
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
import copy

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")

#RESNET
resnet = models.resnet34(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, num_ftrs)
resnet = resnet.to(device)

class LinearSVM(nn.Module):

	def __init__(self):
		super(LinearSVM, self).__init__()
		self.fc = nn.Linear(num_ftrs, 5)  #5 for 5 classes
	def forward(self, x):
		out = self.fc(x)
		return out

SVM = LinearSVM()
SVM.to(device)
optimizer = optim.SGD(SVM.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
batch_size = 4

class LoadDataset(Dataset):
	def __init__(self, phase):
		self.phase = phase
		if phase == 'train':
			self.data = self.readtxt('SVMtrain')
		if phase == 'val':
			self.data = self.readtxt('SVMval')

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		#***Load data***		
		img_name, cls = self.data[idx].split(' ')

		#read images
		img = io.imread(img_name)

		#resize images
		img = resize(img, (224, 224))

		#transform into (3, x, x)
		img = img.transpose(2, 0, 1)

		#convert to torch.tensors
		img = torch.from_numpy(img).float()

		if self.phase == 'train':
			#similar to one hot encoding
			y = torch.zeros(5) #5 for 5 classes
			y.fill_(-1)
			y[int(cls)-1] = 1  #index start from zero, classes start from 1

			sample = {'x':img, 'y':y}

		elif self.phase == 'val':
			sample = {'x':img, 'cls':int(cls)}

		return sample

	def readtxt(self, txtfile):
		f = open(os.getcwd()+'/'+txtfile+'.txt', 'r')
		a = f.read().split('\n')
		for i in range(len(a)):
			a[i].split(' ')
		return a[:-1]

def hingeLoss(out, y):
	hinge = 1 - torch.sum(out*y, dim = 1)
	return torch.max(hinge, torch.Tensor([0]).to(device)).mean() 

def train_model(model, optimizer, scheduler, num_epochs=25):
	since = time.time()
	#weights
	best_model_wts = copy.deepcopy(model.state_dict())
	#accuracies
	best_acc = 0.0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs))
		print('-'*10)
		scheduler.step()
		#Each iteration has a training and validation phase
		for inputs in Dataloader['train']:
			model.train()
			x = inputs['x'].to(device)
			y = inputs['y'].to(device)

			optimizer.zero_grad()

			out = model(resnet(x))

			loss = hingeLoss(out, y)
			loss.backward()
			optimizer.step()

		npreds = 0 #no of predictions
		cpreds = 0 #no of correct predictions
		for inputs in Dataloader['val']:
			npreds += 1
			model.eval()
			x = inputs['x'].to(device)
			cls = inputs['cls']
			with torch.no_grad():
				out = model(resnet(x))

			#find brightest neuron in output mx(max)
			mx = out[0]
			y = 1
			for i in range(1, 5):
				if out[i]>mx:
					mx = out[i]
					y = i+1 #index start from zero, classes start from 1
			if y == cls:
				cpred += 1

		epoch_acc = cpreds/npreds

		if epoch_acc > best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict())
			model.load_state_dict(best_model_wts)
			torch.save(model.state_dict(), os.getcwd()+'/weights/SVM.pt')
		print('epoch_acc:{}\n'.format(epoch_acc))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	return

Dataloader = {'train':DataLoader(LoadDataset('train'), batch_size=batch_size, shuffle=True, num_workers=4), 'val': DataLoader(LoadDataset('val'), batch_size=1, shuffle=True, num_workers=4)}

train_model(SVM, optimizer, scheduler, 100)





