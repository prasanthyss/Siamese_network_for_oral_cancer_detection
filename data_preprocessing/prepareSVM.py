import os
from random import randint

#initialize iterator and sizes for each class
itr = [1]*5
#sz = [134, 1381, 1382, 45, 390] #train
sz = [17, 205, 141, 14, 39] #val

#open train.txt file
f = open(os.getcwd()+'/SVMval.txt', 'w+')


for i in range(1000):
	m = randint(1, 5)

	cwd = os.getcwd()
	#val
	f.write(cwd+'/dataset/val/'+str(m)+'/v'+str(itr[m-1]%sz[m-1]+1)+'.png'+' '+str(m)+'\n')
	#train
	#f.write(cwd+'/dataset/train/'+str(m)+'/t'+str(itr[m-1]%sz[m-1]+1)+'.png'+' '+str(m)+'\n')

	itr[m-1] = itr[m-1]+1
	

	



