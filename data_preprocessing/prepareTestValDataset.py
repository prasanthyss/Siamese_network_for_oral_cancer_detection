import os
from random import randint

#initialize iterator and sizes for each class
itr = [1]*5
sz = [17, 205, 141, 14, 39] #val
#sz = [26, 459, 383, 57, 81] #test

#open train.txt file
f = open(os.getcwd()+'/val.txt', 'w+')

for i in range(1000):
	m = randint(1, 5)
	n = randint(1, 5)
	while m != n: n = randint(1, 5)
	s = 1 if m==n else 0
	cwd = os.getcwd()
	f.write('/home/prasanth/Siamese/dataset/train/'+str(m)+'/t1.png'+' '+ cwd+'/dataset/val/'+str(n)+'/v'+str(sz[n-1]-itr[n-1]%sz[n-1])+'.png'+' '+str(s)+'\n')
	#Update iterators
	itr[m-1] = itr[m-1]+1
	itr[n-1] = itr[n-1]+1

	



