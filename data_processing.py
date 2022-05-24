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
        groups = self.df.groupby('Final Train ID')
        df_re = groups.get_group(1)
        df_up = groups.get_group(27)
        up_groups = df_up.groupby('Next Train ID')
        re_groups = df_re.groupby('Next Train ID')
        up_idxs = list(map(lambda x: 2*x,range(3,29))) #even num. btw 6~56
        re_idxs = list(map(lambda x: 2*x+1,range(1,27))) #odd num. btw 3~53
        for idx in up_idxs:
            _df = up_groups.get_group(idx)
            _dest = os.path.join(dest,"up_" + '{:02d}'.format(idx) + ".csv")
            _df.to_csv(_dest,index=False)
        for idx in re_idxs:
            _df = re_groups.get_group(idx)
            _dest = os.path.join(dest,"re_" + '{:02d}'.format(idx) + ".csv")
            _df.to_csv(_dest,index=False)

    def save_csv_up_re(self,dest):
        groups = self.df.groupby('Final Train ID')
        df_re = groups.get_group(1) #odd num. btw 3~53
        df_re.drop(df_re[df_re['Next Train ID'] == 4].index,inplace=True) #odd num.
        df_re.drop(df_re[df_re['Next Train ID'] == 55].index,inplace=True) #btw 3~53
        df_up = groups.get_group(27)
        df_up.drop(df_up[df_up['Next Train ID'] == 4].index,inplace=True) #btw 6~56
        df_up.drop(df_up[df_up['Next Train ID'] == 55].index,inplace=True) #even num.
        dest_up = os.path.join(dest,"up.csv")
        df_up.to_csv(dest_up,index=False)
        dest_re = os.path.join(dest,"re.csv")
        df_re.to_csv(dest_re,index=False)

    def debug(self):
        print(self.df_num)

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
    car_data_filter_list = ['Time','Next Train ID','Final Train ID',\
        'Distance from station','Train Speed']
    ter_data_filter_list = ['Time','BS_1_1','RSRP_1_1','RSSI_1_1','RSRQ_1_1',\
        'BS_2_1','RSRP_2_1','RSSI_2_1','RSRQ_2_1']

    data_dir = "./data"
    car_data_dir = os.path.join(data_dir,"raw_data/car_logging_data")
    ter_data_dir = os.path.join(data_dir,"raw_data/terminal_logging_data")
    csv_all_save_dir = os.path.join(data_dir,"processed_data/all")
    csv_split_save_dir = os.path.join(data_dir,"processed_data/split")
    csv_up_re_save_dir = os.path.join(data_dir,"processed_data/up_re")
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
        memory.store(mer_df)
    #memory.debug()
    #csv_all_name = os.path.join(csv_all_save_dir,"all.csv")
    #memory.save_csv_all(csv_all_name)
    #memory.save_csv_split_destination(csv_split_save_dir)
    memory.save_csv_up_re(csv_up_re_save_dir)