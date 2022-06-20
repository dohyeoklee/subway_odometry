import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
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
				#self.train_x_data = self.min_max_scaler(x_data,train_idxs)
				self.train_x_data = self.robust_scaler(x_data,train_idxs)
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

	def min_max_scaler(self,x_data,train_idxs):
		_x_train = np.array([x_data[i] for i in train_idxs])
		_x_data = np.array(x_data)
		x_min = np.min(_x_train,axis=0)
		x_max = np.max(_x_train,axis=0)
		x_range = np.reciprocal(x_max - x_min,dtype=float)
		x_min_arr = np.repeat(np.array([x_min]),repeats=_x_data.shape[0],axis=0)
		x_range_arr = np.repeat(np.array([x_range]),repeats=_x_data.shape[0],axis=0)
		x_scaled = np.multiply((_x_data - x_min_arr),x_range_arr)
		return x_scaled.tolist()

	def robust_scaler(self,x_data,train_idxs):
		_x_train = np.array([x_data[i] for i in train_idxs])
		_x_data = np.array(x_data)
		percentile = np.nanpercentile(_x_data,[25,50,75],axis=0)
		x_1q = percentile[0]
		x_med = percentile[1]
		x_3q = percentile[2]
		x_iqr = np.reciprocal(x_3q - x_1q,dtype=float)
		x_med_arr = np.repeat(np.array([x_med]),repeats=_x_data.shape[0],axis=0)
		x_iqr_arr = np.repeat(np.array([x_iqr]),repeats=_x_data.shape[0],axis=0)
		x_scaled = np.multiply((_x_data - x_med_arr),x_iqr_arr)
		#print(x_scaled)
		#x_scaled = np.clip(x_scaled,-1.0,1.0)
		print(x_scaled.shape)
		return x_scaled.tolist()

	def huristic_processing(self,path):
		df = pd.read_csv(path)
		df.drop(['Next Train ID'],axis=1,inplace=True)
		df.drop(['Final Train ID'],axis=1,inplace=True)
		result = df.groupby(['BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1',\
			'BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']).aggregate([np.mean])
		result = self.huristic_processing_equal_space(result)
		input_list = result.index.tolist()
		input_list = list(map(list,input_list))
		label_list = np.expand_dims(result.values,axis=1).tolist()
		return input_list,label_list

	def huristic_processing_equal_space(self,df):
		space = 10
		return df.apply(lambda x: space*int(x/space)+space/2,axis=1)

	def get_sampler(self):
		return self.train_sampler,self.test_sampler

class Mlp(nn.Module):
	def __init__(self,hidden_size):
		super(Mlp,self).__init__()
		self.fc1 = nn.Linear(in_features=8,out_features=hidden_size,bias=True)
		self.fc2 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc3 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc4 = nn.Linear(in_features=hidden_size,out_features=1,bias=True)

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
		error = float(abs(pred[0][0].item()-y_test[0][0].item()))
		error_list.append(error)
	mean_error = np.mean(error_list)
	worst_5_error = np.nanpercentile(error_list,95,axis=0)
	worst_error = np.max(error_list)
	return mean_error,worst_5_error,worst_error

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=4)
        m.bias.data.fill_(0.1)

def loss_fn(pred,target):
	loss = F.smooth_l1_loss(pred,target)
	return loss

def train(seed,target_scenario,result_data):
	data_dir = os.path.join("./data/processed_data/split",target_scenario)

	#default setting
	#batch_size = 32
	#hidden_size = 12
	#lr = 5e-2
	#weight_decay=1e-3

	batch_size = 64
	hidden_size = 24#12
	lr = 5e-2
	weight_decay=1e-3
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
	optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma=0.8)

	num_epoch = 3000

	mean_error_list = []
	worst_error_list = []
	worst_5_error_list = []

	for epoch in tqdm(range(num_epoch + 1),desc="epoch loop",leave = False):
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
				mean_error,worst_5_error,worst_error = test_model(model,test_dataloader,device)
				#print('Epoch {:4d}/{}, mean error: {:.6f}, 95 error: {:.6f}, worst error: {:.6f}'\
				#	.format(epoch,num_epoch,mean_error,worst_5_error,worst_error))
				mean_error_list.append(mean_error)
				worst_5_error_list.append(worst_5_error)
				worst_error_list.append(worst_error)
	min_mean_error = min(mean_error_list)
	min_worst_5_error = min(worst_5_error_list)
	min_worst_error = min(worst_error_list)
	result_str = '{}, {}, {:.6f}, {:.6f}, {:.6f}'.\
	format(target_scenario,seed,min_mean_error,min_worst_5_error,min_worst_error)
	print(result_str)
	result_data = result_str.split(',')

	result_data_list.append(result_data)
	return result_data_list

if __name__ == '__main__':
	result_dir = "./result/mlp_onehot/"
	result_data_list = []

	#seeds = [1991,202205,20220502]
	seeds = [2022,199152,5020348]
	root_dir = "./data"
	data_root_dir = os.path.join(root_dir,"processed_data/split")
	#data_list = os.listdir(data_root_dir)
	data_list = ["re_33.csv"]
	for data_name in tqdm(data_list,desc="data loop"):
		for seed in seeds:
			result_data_list = train(seed,data_name,result_data_list)

	#df = pd.DataFrame(result_data_list,columns=['file name','seed','mean','95 percent','worst'])
	#df.to_csv(result_dir + 'mlp_onehot.csv',index=False)