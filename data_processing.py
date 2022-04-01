import pandas as pd
import os
import time

class df_memory(object):
    def __init__(self):
        self.df_num = 0
    
    def store(self,df):
        if self.df_num == 0:
            self.df = df
            self.df_num += 1
        else:
            self.df = pd.concat([self.df,df])
            self.df_num += 1

    def save_csv_all(self,dest):
        self.df.to_csv(dest,index=False)

    def save_csv_split_destination(self,dest):
        df.to_csv(dest,index=False)

    def debug(self):
        print(self.df_num)
        #print(self.df.count)
        self.df.groupby('Final Train Id')

def df_huristic_cleaning(df):
    #delete row which include NaN
    df.dropna(axis=0,inplace=True)
    #delete row when train was stopped
    df.drop(df[df['Train Speed'] == '0 (km/h)'].index,inplace=True)
    df.drop(df[df['Distance from station'] == '0 (m)'].index,inplace=True)
    #delete row which include garbage data
    df.drop(df[df['Next Train ID'] == '-'].index,inplace=True)
    df.drop(df[df['Final Train ID'] == '-'].index,inplace=True)
    #no need for speed and time data
    df.drop(['Train Speed'],axis=1,inplace=True)
    df.drop(['Time'],axis=1,inplace=True)
    #convert Train ID(Next,Final) as integer without korean string
    df['Next Train ID'] = df['Next Train ID'].apply(lambda x: int(x.split('(')[-1][:-1]))
    df['Final Train ID'] = df['Final Train ID'].apply(lambda x: int(x.split('(')[-1][:-1]))
    #convert Distance as integer without string for unit (m)
    df['Distance from station'] = df['Distance from station'].apply(lambda x: int(x.split()[0]))

    return df

if __name__ == '__main__':
    debug_list = ['Time']
    car_data_filter_list = ['Time','Next Train ID','Final Train ID',\
        'Distance from station','Train Speed']
    ter_data_filter_list = ['Time','BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1',\
        'BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']

    data_dir = "./data"
    car_data_dir = os.path.join(data_dir,"raw_data/car_logging_data")
    ter_data_dir = os.path.join(data_dir,"raw_data/terminal_logging_data")
    csv_all_save_dir = os.path.join(data_dir,"processed_data/all")
    csv_split_save_dir = os.path.join(data_dir,"processed_data/split")
    car_data_list = os.listdir(car_data_dir)

    memory = df_memory()

    for _car_data_name in car_data_list:
        car_data_name = os.path.join(car_data_dir,_car_data_name)
        ter_data_name = os.path.join(ter_data_dir,_car_data_name)
        #read excel file
        car_df = pd.read_excel(car_data_name)[car_data_filter_list]
        ter_df = pd.read_excel(ter_data_name)[ter_data_filter_list]
        #car data: string / terminal data: int64
        car_df = car_df.convert_dtypes().astype({'Time':'string'})
        ter_df = ter_df.convert_dtypes().astype({'Time':'string'})
        #merge two data frame
        mer_df = car_df.merge(ter_df)
        #data cleaning
        mer_df = df_huristic_cleaning(mer_df)
        #print(mer_df) #debug
        memory.store(mer_df)
    memory.debug()
    #csv_all_name = os.path.join(csv_all_save_dir,"all.csv")
    #memory.save_csv_all(csv_all_name)