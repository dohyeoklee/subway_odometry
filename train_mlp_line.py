import pandas as pd
import os
import itertools
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

COM_VAR_NAME = ['BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1','BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']

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
				self.train_x_data = self.min_max_scaler(x_data,train_idxs)
				#self.train_x_data = self.robust_scaler(x_data,train_idxs)
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
		return x_scaled.tolist()

	def huristic_processing(self,path):
		df = pd.read_csv(path)
		ind = df['Final Train ID'][0]
		df.drop(['Final Train ID'],axis=1,inplace=True)

		_offset_df = df.drop(COM_VAR_NAME,axis=1)
		offset_df = _offset_df.groupby(['Next Train ID']).aggregate(np.max)
		key_list = offset_df.index.tolist()
		value_list = offset_df.values.squeeze().tolist()
		ori_offset_dict = dict(zip(key_list,value_list))
		offset_dict = dict(zip(key_list,value_list))
		if ind == 27:
			up_idxs = list(map(lambda x: 2*x,range(4,29))) #even num. btw 6(train start) / 8(offset start)~56
			offset_dict[6] = 0
			for idx in up_idxs:
				offset_dict[idx] = ori_offset_dict[idx-2] + offset_dict[idx-2]
		elif ind == 1:
			re_idxs = list(map(lambda x: 2*x+1,range(25,0,-1))) #odd num. btw 53(train start) / 51(offset start)~3
			offset_dict[53] = 0
			for idx in re_idxs:
				offset_dict[idx] = ori_offset_dict[idx+2] + offset_dict[idx+2]
		else:
			#error
			pass
		self.offset_dict = offset_dict
		df['Distance from station'] = df[['Next Train ID','Distance from station']].apply(\
			lambda x: self.add_offset(x[0],x[1]),axis=1)

		result = df.groupby(['Next Train ID']+COM_VAR_NAME).aggregate([np.mean,np.std])
		result = result.fillna(0.0)['Distance from station']
		result.drop(result[result['std'] > 5].index,inplace=True)
		input_list = result.index.tolist()
		input_list = list(map(list,input_list))
		label_list = result.values.tolist()
		return input_list,label_list

	def add_offset(self,idx,dist):
		offset = self.offset_dict[idx]
		new_dist = offset + dist
		return new_dist

	def get_sampler(self):
		return self.train_sampler,self.test_sampler

class Mlp(nn.Module):
	def __init__(self,hidden_size):
		super(Mlp,self).__init__()
		self.fc1 = nn.Linear(in_features=9,out_features=hidden_size,bias=True)
		self.fc2 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc3 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc4 = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=True)
		self.fc5 = nn.Linear(in_features=hidden_size,out_features=2,bias=True)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(x + self.fc2(x))
		x = F.relu(x + self.fc3(x))
		x = F.relu(x + self.fc4(x))
		return self.fc5(x)

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
	worst_5_error = np.nanpercentile(error_list,95,axis=0)
	worst_error = np.max(error_list)
	return mean_error,worst_5_error,worst_error

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=10)
        m.bias.data.fill_(0.1)

def loss_fn(pred,target):
	loss = F.smooth_l1_loss(pred,target)
	return loss

def train(num_epoch,test,scaling,seed,batch_size,hidden_size,lr):
	root_dir = "./data"
	target_scenario = "up.csv"
	data_dir = os.path.join(root_dir,"processed_data/up_re",target_scenario)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.9)
	#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,\
	#			lr_lambda=lambda epoch: 0.99 ** epoch,last_epoch=-1,verbose=False)

	mean_error_list = []
	worst_error_list = []
	worst_5_error_list = []

	for epoch in range(num_epoch + 1):
		for batch_idx, samples in enumerate(train_dataloader):
			x_train, y_train = samples
			x_train, y_train = x_train.to(device), y_train.to(device)
			pred = model(x_train)
			loss = loss_fn(pred,y_train)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()

		if test:
			with torch.no_grad():
				mean_error,worst_5_error,worst_error = test_model(model,test_dataloader,device)
				#print('Epoch {:4d}/{}, mean error: {:.6f}, 95 percent error: {:.6f}, worst error: {:.6f}'\
				#	.format(epoch,num_epoch,mean_error,worst_5_error,worst_error))
				mean_error_list.append(mean_error)
				worst_5_error_list.append(worst_5_error)
				worst_error_list.append(worst_error)
	min_mean_error = min(mean_error_list)
	min_worst_5_error = min(worst_5_error_list)
	min_worst_error = min(worst_error_list)
	#print('mean error: {:.6f}, 95 percent error: {:.6f}, worst error: {:.6f}'.\
	#	format(min_mean_error,min_worst_5_error,min_worst_error))
	return min_worst_error

if __name__ == '__main__':
	seeds = [1991,202205,20220502]
	batch_space = [16,32,64,128,256]
	hidden_space = [8,9,10,11,12,13]
	lr_space = [i**(-j) for i in range(1,10) for j in [2,3]]
	hyperparam_space = itertools.product(batch_space,hidden_space,lr_space)
	hyperparam_space_list = [item for item in hyperparam_space]
	test = True
	scaling = True
	num_epoch = 2000

	optim_val_score = 1e5

	for batch_size,hidden_size,lr in tqdm(hyperparam_space_list,desc="outer loop"):
		val_score = 0.0
		for seed in tqdm(seeds,desc="inner loop",leave = False):
			val_score = val_score + train(num_epoch,test,scaling,seed,batch_size,hidden_size,lr)
			if val_score < optim_val_score:
				optim_val_score = val_score
				optim_param = [batch_size,hidden_size,lr]
	print(optim_param)