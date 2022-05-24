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
	def __init__(self,path,test=True,scaling=True):
		if test:
			x_data,self.train_y_data = self.huristic_processing(path)
			dataset_size = len(x_data)
			idxs = list(range(dataset_size))
			split = int(np.floor(0.2*dataset_size))
			np.random.shuffle(idxs)
			train_idxs, test_idxs = idxs[split:], idxs[:split]
			self.train_sampler = SubsetRandomSampler(train_idxs)
			self.test_sampler = SubsetRandomSampler(test_idxs)

			if scaling:
				_x_train = np.array([x_data[i] for i in train_idxs])
				_x_data = np.array(x_data)
				x_min = np.min(_x_train,axis=0)
				x_max = np.max(_x_train,axis=0)
				x_range = np.reciprocal(x_max - x_min,dtype=float)
				x_min_arr = np.repeat(np.array([x_min]),repeats=_x_data.shape[0],axis=0)
				x_range_arr = np.repeat(np.array([x_range]),repeats=_x_data.shape[0],axis=0)
				x_scaled = np.multiply((_x_data - x_min_arr),x_range_arr)
				self.train_x_data = x_scaled.tolist()
			else:
				self.train_x_data = x_data
		else:
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
		result = df.groupby(['BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1',\
			'BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']).aggregate([np.mean,np.std])
		result = result.fillna(0.0)['Distance from station']
		#print(len(result[result['std'] > 20].index))
		result.drop(result[result['std'] > 20].index,inplace=True)
		input_list = result.index.tolist()
		input_list = list(map(list,input_list))
		label_list = result.values.tolist()
		return input_list,label_list

	def get_sampler(self):
		return self.train_sampler,self.test_sampler

class Mlp(nn.Module):
	def __init__(self,hidden_size):
		super(Mlp,self).__init__()
		self.fc1 = nn.Linear(in_features=8,out_features=hidden_size,bias=True)
		self.fc2 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc3 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc4 = nn.Linear(in_features=hidden_size,out_features=2,bias=True)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(x + self.fc2(x))
		x = F.relu(x + self.fc3(x))
		return self.fc4(x)

def test_model(model,test_dataloader,device):
	error_list = []
	for batch_idx, samples in enumerate(test_dataloader):
		x_test, y_test = samples
		x_test, y_test = x_test.to(device), y_test.to(device)
		pred = model(x_test)
		pred_infer = pred[0][0].item() + np.random.rand(1)*pred[0][1].item()
		gt_infer = y_test[0][0].item() + np.random.rand(1)*y_test[0][1].item()
		error = float(abs(pred_infer-gt_infer))
		error_list.append(error)
	mean_error = np.mean(error_list)
	max_error = np.max(error_list)
	return mean_error,max_error

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=4)
        m.bias.data.fill_(0.1)

def loss_fn(pred,target):
	loss = F.smooth_l1_loss(pred,target)
	return loss

def train(seed):
	root_dir = "./data"
	target_scenario = "re_03.csv" # using data_list = os.listdir() for all
	data_dir = os.path.join(root_dir,"processed_data/split",target_scenario)

	batch_size = 32
	hidden_size = 12
	lr = 5e-2
	test = True
	scaling = True

	#device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device == 'cuda':
		torch.cuda.manual_seed_all(seed)

	dataset = SubwayDataset(path=data_dir,test=test,scaling=scaling)
	if test:
		train_sampler,test_sampler = dataset.get_sampler()
		train_dataloader = DataLoader(dataset,batch_size=batch_size,sampler=train_sampler)
		test_dataloader = DataLoader(dataset,batch_size=1,sampler=test_sampler)
	else:
		train_dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

	model = Mlp(hidden_size)
	model = model.to(device)
	model.apply(init_weights)
	optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-3) #weight_decay=1e-3
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma=0.8)

	num_epoch = 3000
	error = 0.0
	mean_error_list = []
	max_error_list = []

	for epoch in range(num_epoch + 1):
		for batch_idx, samples in enumerate(train_dataloader):
			x_train, y_train = samples
			x_train, y_train = x_train.to(device), y_train.to(device)
			pred = model(x_train)
			loss = loss_fn(pred,y_train)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print('Epoch {:4d}/{} Batch {}/{} avg. loss: {:.6f}' \
			#	.format(epoch,num_epoch,batch_idx+1,len(train_dataloader), \
			#	loss.item()))
		scheduler.step()

		if test:
			with torch.no_grad():
				mean_error,max_error = test_model(model,test_dataloader,device)
				print('Epoch {:4d}/{}, mean error: {:.6f}, worst error: {:.6f}'\
					.format(epoch,num_epoch,mean_error,max_error))
				mean_error_list.append(mean_error)
				max_error_list.append(max_error)
	min_mean_error = min(mean_error_list)
	min_max_error = min(max_error_list)
	print('min mean error: {:.6f}, min worst error: {:.6f}'.\
		format(min_mean_error,min_max_error))

if __name__ == '__main__':
	seeds = [1991,202205,20220502]
	for seed in seeds:
		train(seed)