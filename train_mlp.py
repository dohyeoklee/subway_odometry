import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class SubwayDataset(Dataset):
	def __init__(self,path):
		self.train_x_data,self.train_y_data = self.huristic_processing(path)

	def __len__(self):
		return len(self.train_x_data)

	def __getitem__(self,idx):
		x = torch.FloatTensor(self.train_x_data[idx])
		y = torch.FloatTensor(self.train_y_data[idx])
		return x,y

	def huristic_processing(self,path):
		df = pd.read_csv(path)
		df.drop(['Next Train ID'],axis=1,inplace=True)
		df.drop(['Final Train ID'],axis=1,inplace=True)
		result = df.groupby(['BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1','BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']).aggregate([np.mean,np.std])
		result = result.fillna(0)['Distance from station']
		input_list = result.index.tolist()
		input_list = list(map(list,input_list))
		label_list = result.values.tolist()
		return input_list,label_list

	def get_test_data(self):
		pass

	def debug(self):
		pass

class Mlp(nn.Module):
	def __init__(self,hidden_size):
		super(Mlp,self).__init__()
		self.fc1 = nn.Linear(in_features=8,out_features=hidden_size,bias=True)
		self.fc2 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc3 = nn.Linear(in_features=hidden_size,out_features=4,bias=True)
		self.fc4 = nn.Linear(in_features=4,out_features=2,bias=False)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)

if __name__ == '__main__':
	root_dir = "./data"
	target_scenario = "re_03.csv" # using data_list = os.listdir() for all
	data_dir = os.path.join(root_dir,"processed_data/split",target_scenario)

	seed = 20220502
	batch_size = 32
	hidden_size = 12
	lr = 5e-2
	test = True

	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	np.random.seed(seed)
	torch.manual_seed(seed)
	#if device == 'cuda':
	#	torch.cuda.manual_seed_all(seed)

	dataset = SubwayDataset(path=data_dir)
	if test:
		dataset_size = len(dataset)
		idxs = list(range(dataset_size))
		split = int(np.floor(0.2*dataset_size))
		np.random.shuffle(idxs)
		train_idxs, test_idxs = idxs[split:], idxs[:split]
		train_sampler = SubsetRandomSampler(train_idxs)
		test_sampler = SubsetRandomSampler(test_idxs)
		train_dataloader = DataLoader(dataset,batch_size=batch_size,sampler=train_sampler)
		test_dataloader = DataLoader(dataset,batch_size=1,sampler=test_sampler)
	else:
		train_dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

	model = Mlp(hidden_size)
	# add weight initialization
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)

	num_epoch = 200
	for epoch in range(num_epoch + 1):
		for batch_idx, samples in enumerate(train_dataloader):
			x_train, y_train = samples
			pred = model(x_train)
			loss = F.mse_loss(pred,y_train)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('Epoch {:4d}/{} Batch {}/{} avg. loss: {:.6f}' \
				.format(epoch,num_epoch,batch_idx+1,len(train_dataloader), \
				loss.item()))

	if test:
		'''
		root_dir = "./data"
		target_scenario = "re_05.csv" # using data_list = os.listdir() for all
		data_dir = os.path.join(root_dir,"processed_data/split",target_scenario)
	
		dataset = SubwayDataset(path=data_dir)
		dataset_size = len(dataset)
		idxs = list(range(dataset_size))
		split = int(np.floor(0.2*dataset_size))
		np.random.shuffle(idxs)
		test_idxs = idxs[:split]
		test_sampler = SubsetRandomSampler(test_idxs)
		test_dataloader = DataLoader(dataset,batch_size=1,sampler=test_sampler)
		'''

		error = 0.0
		for batch_idx, samples in enumerate(test_dataloader):
			x_test, y_test = samples
			pred = model(x_test)
			#loss = F.mse_loss(pred,y_train)
			error = error + torch.abs(pred[0][0]-y_test[0][0]).item()
		error = error / len(test_dataloader)
		print('avg. loss: {:.6f}'.format(error))
