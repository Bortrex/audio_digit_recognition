import sys
import os
import shutil

import numpy as np
import pandas as pd

if __name__ == "__main__":
	n_coefficients = 13

	path = 'wav_train/'
	os.system('python wav2mfcc.py -c {} -p {}'.format(n_coefficients,path))

	if not os.path.exists('mfcc_train'):
		os.mkdir('mfcc_train')
	for o in os.listdir(path):
		if o[-4:] == "mfcc":
			shutil.move(path+o,'mfcc_train/'+o)


	path = 'mfcc_train/'
	list_files = os.listdir(path)
	n_samples = len(list_files)
	X_train = np.ones((n_samples, 13*32)) * -9999999

	for o in list_files:
		idx = int(o.split('_')[0].split('-')[1])
		
		if o[-4:] == 'mfcc':
			data = pd.read_csv(path+o)
			n_col = int(data.columns[1].split('.')[0])

			if n_col < 32:
				print('Need to fill with unknown values for sample {}'.format(idx))
			elif n_col > 32:
				print('32 is too small')

			# data = data.as_matrix()
			data = data.values

			c = 0
			for j in range(n_col):
				for i in range(n_coefficients):
						X_train[idx,c] = data[i,j]
						c += 1

	np.save('X_train.npy',np.hstack([np.arange(n_samples).reshape(-1,1),X_train]))

	# path = 'wav_test/'
	#
	# os.system('python wav2mfcc.py -c {} -p {}'.format(n_coefficients, path))
	#
	# if not os.path.exists('mfcc_test'):
	# 	os.mkdir('mfcc_test')
	# for o in os.listdir(path):
	# 	if o[-4:] == "mfcc":
	# 		shutil.move(path+o,'mfcc_test/'+o)
	#
	# path = 'mfcc_test/'
	# list_files = os.listdir(path)
	# n_samples = len(list_files)
	# X_test = np.ones((n_samples, 13*32)) * -9999999
	#
	# for o in list_files:
	# 	idx = int(o.split('_')[0].split('-')[1])
	# 	if o[-4:] == 'mfcc':
	# 		data = pd.read_csv(path+o)
	# 		print(idx)
	# 		n_col = int(data.columns[1].split('.')[0])
	#
	# 		if n_col < 32:
	# 			print('Need to fill with unknown values for sample {}'.format(idx))
	# 		elif n_col > 32:
	# 			print('32 is too small')
	#
	# 		data = data.as_matrix()
	#
	# 		c = 0
	# 		for j in range(n_col):
	# 			for i in range(n_coefficients):
	# 					X_test[idx,c] = data[i,j]
	# 					c += 1
	#
	# np.save('X_test.npy',np.hstack([np.arange(n_samples).reshape(-1,1),X_test]))
