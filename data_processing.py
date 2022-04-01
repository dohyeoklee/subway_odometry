import pandas as pd
import os
import time

debug_list = ['Time']
car_data_filter_list = ['Time','Next Train ID','Final Train ID',\
    'Distance from station','Train Speed']
ter_data_filter_list = ['Time','BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1',\
    'BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']

data_dir = "./data/"
car_data_dir = os.path.join(data_dir,"car_logging_data")
ter_data_dir = os.path.join(data_dir,"terminal_logging_data")
car_data_list = os.listdir(car_data_dir)

for _car_data_name in car_data_list:
    #start = time.time()
    car_data_name = os.path.join(car_data_dir,_car_data_name)
    ter_data_name = os.path.join(ter_data_dir,_car_data_name)
    #car_df = pd.read_excel(car_data_name)[car_data_filter_list]
    #ter_df = pd.read_excel(ter_data_name)[ter_data_filter_list]
    car_df = pd.read_excel(car_data_name)[debug_list]
    ter_df = pd.read_excel(ter_data_name)[debug_list]
    print(car_df.loc[0].dtypes)
    print(ter_df.loc[0].dtypes)
    con_df = car_df.merge(ter_df)
    print(con_df)
    #print(ter_df)
    #print("processing time: ",time.time()-start)