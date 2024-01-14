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

#Load the model and set it for fineTuning
#VGG
vgg = models.vgg16()

vgg.classifier = nn.Sequential(
		nn.Linear(512 * 7 * 7, 4096),
		nn.ReLU(True),
		nn.Linear(4096, 4096),
		nn.Sigmoid(),
		nn.Linear(4096, 100),
	)
vgg.to(device)
#RESNET
resnet = models.resnet34(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, num_ftrs)
resnet = resnet.to(device)

batch_size = 64
training_threshold=0.25
val_threshold=0

optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def tripletLoss(a,p,n,bias):   # Anchor Positive Negative Bias
	normap = F.sigmoid(torch.sum((a-p)*(a-p), dim=1).sqrt())
	norman = F.sigmoid(torch.sum((a-n)*(a-n), dim=1).sqrt())
	return torch.max(normap-norman+bias, torch.Tensor([0]).to(device)).mean()


class LoadDataset(Dataset):
	def __init__(self, phase):
		if phase == 'train':
			self.data = self.readtxt('trainTL')
		else:
			self.data = self.readtxt(phase)
		self.phase = phase
	
	def __len__(self):
			return len(self.data)

	def __getitem__(self, idx):
		if self.phase == 'train':

			#***Load training data***
			a_name, p_name, n_name = self.data[idx].split(' ')
			#read images
			a_img = io.imread(a_name)
			p_img = io.imread(p_name)
			n_img = io.imread(n_name)
			#resize images
			a_img = resize(a_img, (224, 224))
			p_img = resize(p_img, (224, 224))
			n_img = resize(n_img, (224, 224))
			#transform into (3, x, x)
			a_img = a_img.transpose(2, 0, 1)
			p_img = p_img.transpose(2, 0, 1)
			n_img = n_img.transpose(2, 0, 1)
			#convert to torch.tensors
			a_img = torch.from_numpy(a_img).float()
			p_img = torch.from_numpy(p_img).float()
			n_img = torch.from_numpy(n_img).float()

			sample = {'a':a_img, 'p':p_img, 'n':n_img}

		else:
			#***Load validation data***
			im1_name, im2_name, labels = self.data[idx].split(' ')
			#read images
			img1 = io.imread(im1_name)
			img2 = io.imread(im2_name)
			#resize images
			img1 = resize(img1, (224, 224))
			img2 = resize(img2, (224, 224))
			#transform into (3, x, x)
			img1 = img1.transpose(2, 0, 1)
			img2 = img2.transpose(2, 0, 1)
			#convert to torch.tensors
			img1 = torch.from_numpy(img1).float()
			img2 = torch.from_numpy(img2).float()
			sample = {'img1':img1, 'img2':img2, 'label':int(labels)}
		return sample

	def readtxt(self, txtfile):
		f = open(os.getcwd()+'/'+txtfile+'.txt', 'r')
		a = f.read().split('\n')
		for i in range(len(a)):
			a[i].split(' ')
		return a[:-1]

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
		#Each itertion has training and validation phase
		for inputs in Dataloader['train']:
			model.train()
			a = inputs['a'].to(device)
			p = inputs['p'].to(device)
			n = inputs['n'].to(device)
				
			optimizer.zero_grad()

			# forward
			# track history only while training
			a_out = model(a)
			p_out = model(p)
			n_out = model(n)
				
			loss = tripletLoss(a_out, p_out, n_out, training_threshold)
			loss.backward()
			optimizer.step()

		npreds = 0 #no of predictions
		cpreds = 0 #no of correct predictions
		for inputs in Dataloader['val']:
			npreds += 1
			model.eval()
			f = inputs['img1'].to(device)   #feature 
			t = inputs['img2'].to(device)   #target
			lbl = inputs['label']
			with torch.no_grad():
				f_out = model(f)
				t_out = model(t)
			dstnc = F.sigmoid(torch.sum((f_out-t_out)*(f_out-t_out), dim=1).sqrt())
			result = 1 if dstnc<0.8 else 0
			if result == lbl:
				cpreds += 1

		epoch_acc = cpreds/npreds

		if epoch_acc > best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict())
			model.load_state_dict(best_model_wts)
			torch.save(model.state_dict(), os.getcwd()+'/weights/Siamese.pt')
		print('epoch_acc:{}\n'.format(epoch_acc))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	return

#Visualizing the network
##Next work
def visualize_model(model, threshold, test, batchsize=16, featuresize=4096):
	for j in range(10):
		inputs, labels = LoadDataset('test', j, test)
		f = inputs[:,0]
		t = inputs[:,1]
		f1 = model(f)
		t1 = model(t)
		result = open(os.getcwd()+'result.txt', 'a')
	
		for i in range(batchsize):                           #iterate for each test
			distance = ((f1[i]-t1[i])**2).mean().sqrt().item()            
			if(distance <= threshold):
				result.write('same'+'        '+str(labels[i])+'\n')
			else:
				result.write('different'+'   '+str(labels[i])+'\n')



Dataloader = {'train':DataLoader(LoadDataset('train'), batch_size=batch_size, shuffle=True, num_workers=4), 'val': DataLoader(LoadDataset('val'), batch_size=1, shuffle=True, num_workers=4) }


#plt.ion()
#plt.xlabel('iterations')
#plt.ylabel('accuracy')

#Train model and save  weights
train_model(resnet, optimizer, scheduler, 100)
#resnet = train_model(resnet, optimizer, scheduler)


