import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data_dir = "./data"
	csv_save_dir = os.path.join(data_dir,"processed_data/split")
	data_list = os.listdir(csv_save_dir)

	for _data_list in data_list[0:2]:
		data_name = os.path.join(csv_save_dir,_data_list)
		print(data_name)
		df = pd.read_csv(data_name)
		df.drop(['Next Train ID'],axis=1,inplace=True)
		df.drop(['Final Train ID'],axis=1,inplace=True)
		result = df.groupby(['BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1','BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']).aggregate([np.mean,np.std])
		result = result.fillna(0)['Distance from station']
		
		plot_df = pd.DataFrame(result.values,columns=['means','stds'])
		plot_df_sort = plot_df.sort_values(by=['means'])
		plot_df_sort.reset_index(inplace=True)
		df_means = plot_df_sort['means']
		df_stds = plot_df_sort['stds']
		print(df_stds.mean())
		
		plt.plot(df_means.index, df_means.values)
		plt.fill_between(df_means.index, (df_means - df_stds).values, (df_means + df_stds).values,alpha=0.8)
		plt.show()