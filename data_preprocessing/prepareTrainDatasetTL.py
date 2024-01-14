import os
from random import randint

#initialize iterator and sizes for each class
itr = [1]*5
sz = [134, 1381, 1382, 45, 390]

#open train.txt file
f = open(os.getcwd()+'/train.txt', 'w+')
#po = open(os.getcwd()+'/positive.txt', 'w+')
#ne = open(os.getcwd()+'/negative.txt', 'w+')

for i in range(6000):
	m = randint(1, 5)
	n = randint(1, 5)
	p = randint(1, sz[m-1])
	while p <= itr[m-1]%sz[m-1]:                 #Take images which are not taken before
		p = randint(1, sz[m-1]) 
	while m==n:                                  #Negative is different from Anchor
		n = randint(1, 5)
	cwd = os.getcwd()
	f.write(cwd+'/dataset/train/'+str(m)+'/t'+str(itr[m-1]%sz[m-1]+1)+'.png'+' '+cwd+'/dataset/train/'+str(m)+'/t'+str(p)+'.png'+' '+cwd+'/dataset/train/'+str(n)+'/t'+str(itr[n-1]%sz[n-1]+1)+'.png'+'\n')

	#Update iterators
	itr[m-1] = itr[m-1]+1
	itr[n-1] = itr[n-1]+1

	



