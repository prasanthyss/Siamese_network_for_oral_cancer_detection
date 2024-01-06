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

#Load the model and set it for TransferLearning
vgg = models.vgg16(pretrained = True)

vgg.classifier = nn.Sequential(
		nn.Linear(512 * 7 * 7, 4096),
		nn.ReLU(True),
		nn.Linear(4096, 4096)
	)

vgg.to(device)

#RESNET
resnet = models.resnet18(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.to(device)

optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)
batch_size = 64
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def dotprod(a, p):
	norma = torch.sum(a*a, dim=1).sqrt()
	normp = torch.sum(p*p, dim=1).sqrt()
	dotap = torch.sum(a*p, dim=1)/(norma*normp)
	return (dotap+1)/2

class LoadDataset(Dataset):

	def __init__(self, phase):
		self.phase = phase
		if phase == 'train':
			self.data = self.readtxt('trainCE')
		else:
			self.data = self.readtxt(phase)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if self.phase == 'train':
			#***Load training data***
			train1_name, train2_name, target, weight = self.data[idx].split(' ')
			target = torch.Tensor([int(target)])
			weight = torch.Tensor([float(weight)])
			#read images
			train1_img = io.imread(train1_name)
			train2_img = io.imread(train2_name)
			#resize images
			train1_img = resize(train1_img, (224, 224))
			train2_img = resize(train2_img, (224, 224))
			#transform into (3, x, x)
			train1_img = train1_img.transpose(2, 0, 1)
			train2_img = train2_img.transpose(2, 0, 1)
			#convert to torch.tensors
			train1_img = torch.from_numpy(train1_img).float()
			train2_img = torch.from_numpy(train2_img).float()
			sample ={'train1':train1_img,'train2':train2_img,'target':target, 'weight':weight}

		else:
			#***Load validation data***
			val1_name, val2_name, labels = self.val_data[idx].split(' ')
			#read images
			val1 = io.imread(val1_name)
			val2 = io.imread(val2_name)
			#resize images
			val1 = resize(val1, (224, 224))
			val2 = resize(val2, (224, 224))
			#transform into (3, x, x)
			val1 = val1.transpose(2, 0, 1)
			val2 = val2.transpose(2, 0, 1)
			#convert to torch.tensors
			val1 = torch.from_numpy(val1).float()
			val2 = torch.from_numpy(val2).float()
			sample ={'val1':val1, 'val2':val2, 'label':int(labels)}
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
	predictions = np.empty(0) #store last 100 predictions
	acc = 0 #accuracy of model
	accList = np.empty(0) #store accs
	lossList = np.empty(0)
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs))
		print('-'*10)
		
		#Each itertion has training and validation phase
		for inputs in dataloader['train']:

			scheduler.step()
			train1 = inputs['train1'].to(device)
			train2 = inputs['train2'].to(device)
			target = inputs['target'].to(device)  #1D Tensor
			weight = inputs['weight'].to(device)
			
			optimizer.zero_grad()

			# forward
			train1_out = model(train1)
			train2_out = model(train2)

			output = dotprod(train1_out, train2_out)
			loss = F.binary_cross_entropy(output, target)
			loss.backward()
			optimizer.step()

			lossList = np.append(lossList, np.array([loss.item()]))
			plt.plot(lossList, 'b')
			plt.pause(0.05)

		npreds = 0 #no of predictions
		cpreds = 0 #no of correct predictions
		for inputs in dataloader['val']:
			npreds += 1
			model.eval()
			f = inputs['val1'].to(device)   #feature 
			t = inputs['val2'].to(device)   #target
			lbl = inputs['label']
			with torch.no_grad():
				f_out = model(f)
				t_out = model(t)
			result = 1 if dotprod(f_out, t_out).item()>0.5 else 0
			if result == lbl:
				cpreds += 1

		epoch_acc = cpreds/npreds

		if epoch_acc > best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict)
			torch.save(model.state_dict(), os.getcwd()+'/weights/Siamese:{}.wts'.format(epoch))

		print('epoch_acc:{}\n'.format(epoch_acc))
		
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))



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



dataloader  = { 'train' : DataLoader(LoadDataset('train'), batch_size=batch_size, shuffle=True, num_workers=4), 'val' : DataLoader(LoadDataset('val'), batch_size=1, shuffle=True, num_workers=4)}

plt.ion()
plt.xlabel('iterations')
plt.ylabel('loss')

#Train model 
train_model(resnet, optimizer, scheduler)

