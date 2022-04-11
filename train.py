import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SubwayDataset(Dataset):
	def __init__(self,path,train=True):
		self.x_data,self.y_data = self.huristic_processing(path)
		if train:
			pass

	def __len__(self):
		return len(self.x_data)

	def __getitem__(self,idx):
		x = torch.FloatTensor(self.x_data[idx])
		y = torch.FloatTensor(self.y_data[idx])
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

	def debug(self):
		pass

class Mlp(nn.Module):
	def __init__(self,hidden_size):
		super(Mlp,self).__init__()
		self.fc1 = nn.Linear(in_features=8,out_features=hidden_size,bias=False)
		self.fc2 = nn.Linear(in_features=hidden_size,out_features=2,bias=False)

	def forward(self,x):
		x = self.fc1(x)
		return self.fc2(x)

if __name__ == '__main__':
	root_dir = "./data"
	target_scenario = "re_03.csv" # using data_list = os.listdir() for all
	data_dir = os.path.join(root_dir,"processed_data/split",target_scenario)

	seed = 20220411
	batch_size = 8
	hidden_size = 8

	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	np.random.seed(seed)
	torch.manual_seed(seed)
	#if device == 'cuda':
	#	torch.cuda.manual_seed_all(seed)

	dataset = SubwayDataset(path=data_dir,train=True)	
	dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

	model = Mlp(hidden_size)
	# add weight initialization
	optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

	num_epoch = 100
	for epoch in range(num_epoch + 1):
		for batch_idx, samples in enumerate(dataloader):
			x_train, y_train = samples
			pred = model(x_train)
			loss = F.mse_loss(pred,y_train)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('Epoch {:4d}/{} Batch {}/{} avg. loss: {:.6f}' \
				.format(epoch,num_epoch,batch_idx+1,len(dataloader), \
				loss.item()))
