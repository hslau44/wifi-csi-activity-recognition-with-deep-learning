import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



def radio_image(data):
    plt.figure(figsize=(20,8))
    plt.imshow(data.T,interpolation = "nearest", aspect = "auto", cmap="jet")
    return

def radio_image_fp(fp,rgn=[0,90],scaler=MinMaxScaler):
    a,b = rgn
    data = pd.read_csv(fp).to_numpy()[:,a:b]
    if scaler:
        data = scaler().fit_transform(data)
    radio_image(data)
    return

def reduce_plot(df,factor=1,mk=None):
    for user_ in df['user'].unique():
        print(user_)
        df_ = df[df['user'] == user_]
        reducer = umap.UMAP(verbose=False)
        reducer.fit(df_.iloc[:,:-2].sample(frac=factor).to_numpy())
        data = reducer.transform(df_.iloc[:,:-2].to_numpy())

        lbs = df_.iloc[:,-1].to_numpy()
        points = diff_marker(lbs)

        fig = plt.figure(figsize = (20,10))
        for i in range(data.shape[-1]):
            plt.plot(data[:,i])
        for p in points:
            plt.axvline(x=p,c='g')

        plt.show()
        if mk != None:
            fp = f'pic/test/{user_}_umap_mk{mk}.png'
            fig.savefig(fp)  
        
        del df_, data, lbs
                    
    return


def csi_data_plot(df,):
    for user_ in df['user'].unique():
        print(user_)

        data = df[df['user'] == user_].iloc[:,:-2].to_numpy()
        reducer = umap.UMAP()
        reducer.fit(data.sample(frac=1)
        data = reducer.transform(data)

        lbs = df[df['user'] == user_].iloc[:,-1].to_numpy()
        points = diff_marker(lbs)

        fig = plt.figure(figsize = (20,10))
        for i in range(reduced_data.shape[-1]):
            plt.plot(data[:,i])

        for p in points:
            plt.axvline(x=p,c='g')

        plt.show()
        if sv == True
            fp = f'pic/test/{user_}_umap_t{count}.png'
            fig.savefig(fp)                
    return 
       
                    
                    
def plot_subplot(X,y,number=10,col=2):
    if number%col > 0:
        row = number//col+1
    else:
        row = number//col
    fig, axs = plt.subplots(row, col, figsize=(int(col*5),int(row*2)))
    for i in range(number): 
        axs[i//col, i%col].imshow(X[i].transpose(1,0,2))
        axs[i//col, i%col].set_title(f'{y[i]}_sample_{i}')
    return