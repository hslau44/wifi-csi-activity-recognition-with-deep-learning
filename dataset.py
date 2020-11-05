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
        
    def __init__(self):
        self.data = []
        self.encoders = np.empty(shape=(1,3),dtype=np.ndarray)
    
    def import_data(self,features_ls,labels_ls,window_size,slide_size,skip_labels=None):
#         self.data = []
#         assert len(features_ls) == len(labels_ls)
#         for item_idx,(features,labels) in enumerate(zip(features_ls,label_ls)):
#             X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
#             z = np.full_like(y,item_idx)
#             assert X.shape[0] == y.shape[0] == z.shape[0]
#             self.data.append([X,y,z])
#             print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
#         self.data = np.array(self.data,dtype=object)
#         print('size of DatasetObject ------ : ',self.data.shape) 
        
        assert len(features_ls) == len(labels_ls)
        self.data = np.empty(shape=(len(features_ls),3),dtype=np.ndarray)
        for item_idx,(features,labels) in enumerate(zip(features_ls,labels_ls)):
            X,y = utils.slide_augmentation(features,labels,window_size,slide_size,skip_labels)
            z = np.full_like(y,item_idx)
            assert X.shape[0] == y.shape[0] == z.shape[0]
            self.data[item_idx,0] = X
            self.data[item_idx,1] = y
            self.data[item_idx,2] = z
            print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
        print('size of DatasetObject ------ : ',self.data.shape) 
        return 
        
    def __call__(self):
        X  = np.concatenate(self.data[:,0],axis=0)
        y  = np.concatenate(self.data[:,1],axis=0)
        z  = np.concatenate(self.data[:,2],axis=0)
        return X,y,z
    
    def data_transform(self,func,axis=0,col=None):
        if axis == 0:
            for i in range(self.data.shape[0]):
                self.data[i,0],self.data[i,1],self.data[i,2] = func(self.data[i,0],self.data[i,1],self.data[i,2])
        elif axis == 1:
            for i in range(self.data.shape[0]):
                self.data[i,col] = func(self.data[i,col])
        else:
            print('No transformation is made')
        return
    
    def shape(self):
        for i in range(self.data.shape[0]):
            print(f'index {i} arrays sizes ------ X: ',self.data[i,0].shape,' Y: ',
                  self.data[i,1].shape,' Z: ',self.data[i,2].shape)
        print('size of DatasetObject ------ : ',self.data.shape) 
        return 
    
    def query(self,testset):
        q = [i for i in range(self.data.shape[0])]
        for t in testset:
            q.remove(t)
        print('train set:',q,'\ttest set:',testset)
        X_train = np.concatenate(self.data[q,0],axis=0)
        y_train = np.concatenate(self.data[q,1],axis=0)
        z_train = np.concatenate(self.data[q,2],axis=0)
        X_test  = np.concatenate(self.data[testset,0],axis=0)
        y_test  = np.concatenate(self.data[testset,1],axis=0)
        z_test  = np.concatenate(self.data[testset,2],axis=0)
        del q
        return  [X_train,y_train,z_train],[X_test,y_test,z_test]
    
    def label_encode(self,col=1,encoder=OneHotEncoder()):
        if self.encoders[0,col] != None:
            print('array encoded')
            return 
        self.encoders[0,col] = encoder
        self.encoders[0,col].fit(np.concatenate(self.data[:,col],axis=0).reshape(-1,1))
        for i in range(self.data.shape[0]):
            self.data[i,col] = self.encoders[0,col].transform(self.data[i,col].reshape(-1,1)).toarray()
        return encoder