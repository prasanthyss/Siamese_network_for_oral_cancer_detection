import os
from random import randint

#initialize iterator and sizes for each class
itr = [1]*5
sz = [134, 1381, 1382, 45, 390]
n_data = 3332 #total number of images

#open train.txt file
cwd=os.getcwd()
f = open(cwd+'/trainCE.txt', 'w+')

for i, size in enumerate(sz):
	for j in range(3000):
		m = randint(1, size)
		n = randint(1, size)
		m = size+1-n if m==n else m
		f.write(cwd+'/dataset/train/'+str(i+1)+'/t'+str(m)+'.png'+' '+ cwd+'/dataset/train/'+str(i+1)+'/t'+str(n)+'.png'+' '+str(1)+' '+str(n_data/size)+'\n')

		c = randint(0, 4) 
		while i == c:
			c = randint(0, 4) 
		m = randint(1, size)
		n = randint(1, sz[c])
		wt = i if size<sz[c] else c
		f.write(cwd+'/dataset/train/'+str(i+1)+'/t'+str(m)+'.png'+' '+ cwd+'/dataset/train/'+str(c+1)+'/t'+str(n)+'.png'+' '+str(0)+' '+str(n_data/sz[wt])+'\n')


		
		


