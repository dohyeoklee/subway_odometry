import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data_dir = "./data"
	csv_save_dir = os.path.join(data_dir,"processed_data/split")
	data_list = os.listdir(csv_save_dir)

	for _data_list in data_list:
		data_name = os.path.join(csv_save_dir,_data_list)
		print(data_name)
		df = pd.read_csv(data_name)
		df.drop(['Next Train ID'],axis=1,inplace=True)
		df.drop(['Final Train ID'],axis=1,inplace=True)
		result = df.groupby(['BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1','BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']).aggregate([np.mean,np.std])
		result = result.fillna(0)['Distance from station']
		means = result['mean']
		stds = result['std']
		datas_list = []
		for i in range(means.shape[0]):
			datas_list.append([means.iloc[i],stds.iloc[i]])
		plot_df = pd.DataFrame(datas_list,columns=['means','stds'])
		#print(plot_df)
		
		sns.lineplot(data=plot_df)
		plt.show()