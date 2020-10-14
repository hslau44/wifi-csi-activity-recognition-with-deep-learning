import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle
import utils

def generate_filepaths(fp):
    return fp+"/input*.csv", fp+"/annotation*.csv"

def col_generate(exp_no):
    if exp_no == 'exp_1':
        col = []
        for i in range(1,91):
            col.append(str(f'amp_{i}'))
        for i in range(1,91):
            col.append(str(f'theta_{i}'))
    elif exp_no == 'exp_2':
        col = ['time','unknown_1']
        for i in range(1,91):
            col.append(str(f'amp_{i}'))
    else:  raise Exception("Error occured in function: col_generate")
    return col

def import_single_file(filepath_x,filepath_y,sample_id,exp_no):
    df = pd.read_csv(filepath_x,names=col_generate(exp_no),header=0)
    df['user'] = sample_id
    df['label'] = pd.read_csv(filepath_y,names=['label'],header=0)
    return df

def import_dataframe(fp):
    searchpaths_x,searchpaths_y = generate_filepaths(fp)
    exp_no = fp.split("/")[-1]
    dataframes = []
    filepaths_x = sorted(glob.glob(searchpaths_x))
    filepaths_y = sorted(glob.glob(searchpaths_y))
    assert len(filepaths_x) == len(filepaths_y)
    print(f"Found {len(filepaths_x)} files.")
    for filepath_x,filepath_y in zip(filepaths_x,filepaths_y):
        user = filepath_x.split("/")[-1].split("_")[1]
        dataframes.append(import_single_file(filepath_x,filepath_y,user,exp_no))
        print(filepath_x.split("/")[-1],filepath_y.split("/")[-1],user)
    return pd.concat(dataframes,axis=0)


class DatasetObject:
        
    def __init__(self,df,window_size=1000,slide_size=200,scaling=True,resampling=True,cnn=True,stacking=False):
        self.data = []
        dataframes = [df[df.user == i] for i in df.user.unique()]
        for df in dataframes:
            if scaling == True:
                df,_ = utils.scale_dataframe(df,MinMaxScaler(),True)
            X,y,z = utils.data_transform(df,window_size,slide_size,impurity=0,skp=True,flatten=False,reshape=True)
            if resampling == True:
                s = utils.resampling(X,y,False,True)
                X,y = shuffle(X[s],y[s])
            if cnn == True:
                X = X.reshape(-1,window_size,90,1)
                if stacking == True:
                    X = 255*X
                    X = np.concatenate((X,X,X),axis=3)
            print(X.shape,y.shape,z.shape)
            self.data.append([X,y,z])
        self.data = np.array(self.data,dtype=object)
        print(self.data.shape) 

    
    def query(self,testset):
        q = [i for i in range(self.data.shape[0])]
        for t in testset:
            q.remove(t)
        print(q)
        X_train = np.concatenate(self.data[q,0],axis=0)
        y_train = np.concatenate(self.data[q,1],axis=0)
        z_train = np.concatenate(self.data[q,2],axis=0)
        X_test  = np.concatenate(self.data[testset,0],axis=0)
        y_test  = np.concatenate(self.data[testset,1],axis=0)
        z_test  = np.concatenate(self.data[testset,2],axis=0)
        del q
        return  (X_train,y_train,z_train),(X_test,y_test,z_test)
    
    def one_hot(self,col,ohe=None):
        if ohe == None:
            ohe=OneHotEncoder()
            ohe.fit(np.concatenate(self.data[:,col],axis=0))
        for i in range(self.data.shape[0]):
            self.data[i,col] = ohe.transform(self.data[i,col]).toarray()
        return ohe