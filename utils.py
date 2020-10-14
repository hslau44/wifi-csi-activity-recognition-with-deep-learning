import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

def major_vote(ls,impurity=0.01):
    counter = []
    for category in set(ls):
        c = 0
        for item in ls:
            if item == category:
                c += 1
                continue
        counter.append(c)
    if float(min(counter)) > impurity*sum(counter):
        idx = min(counter)
    else:
        idx = max(counter)
    result = list(set(ls))[counter.index(idx)]
    return result


def data_transform(df,window_size,slide_size,impurity=0.001,skp=False,flatten=False,reshape=False):
    X,y,z = [],[],[]
    for u in df['user'].unique():
        df_ = df[df['user'] == u].reset_index(drop=True)
        for i in range(0,len(df_)-window_size+1,slide_size):
            lb = major_vote(df_.iloc[i:i+window_size,-1],impurity)          
            if (skp == True) & (lb == 'noactivity'):
                continue
            else:
                X.append(df_.iloc[i:i+window_size,:-2].to_numpy())
                z.append(major_vote(df_.iloc[i:i+window_size,-2]))
                y.append(lb)
        del df_
        print(f'{u} done')

    X = np.array(X)
    y = np.array(y)
    z = np.array(z)
    print('done')
    
    if flatten == True:
        X = X.reshape(len(X),-1) 

    if reshape == True:
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
    
    return X,y,z

def stacking(data,scale=255):
    if scale != False:
        scaler = MinMaxScaler()
        data_s = scale*scaler.fit_transform(data.reshape(len(data),-1))
    else:
        data_s = data
    data_s = data_s.reshape(data_s.shape[0],-1,90,1) # change axis 1
    return np.concatenate((data_s,data_s,data_s),axis=3)

def breakpoints(ls):
    points = []
    for i in range(len(ls)-1):
        if ls[i+1] != ls[i]:
            points.append(i)
    return points

def resampling(X,y,oversampling=False,return_idx=False):
    assert len(X) == len(y)
    df = pd.DataFrame(y,columns=['label'])
    if oversampling == True:
        num = max(df['label'].value_counts())
        replace = True
    else: 
        num = min(df['label'].value_counts())
        replace = False
    idx_ls = []
    for c in df['label'].unique():
        idx_c = df[df['label'] == c].sample(num,replace=replace,random_state=0).index
        idx_ls.append(idx_c)
    idx_ls = np.array(idx_ls).reshape(-1,)
    if return_idx == True:
        return idx_ls
    return X[idx_ls],y[idx_ls]

def apply_sampling(item_ls,idx_ls):
    new = []
    for item in item_ls:
        new.append(item[idx_ls])
    return new

def scale_dataframe(df,scaler,fit=False):
    cols = df.columns[:-2]
    data = df.iloc[:,:-2].to_numpy()
    if fit == True:
        print("fit_dataframe")
        data = scaler.fit_transform(data)
    else:
        print("transform_dataframe")
        data = scaler.transform(data)
    return pd.concat([pd.DataFrame(data,columns=cols).reset_index(drop=True),
                      df.iloc[:,-2:].reset_index(drop=True)],axis=1),scaler

def evaluation(model,X_test,y_test,ohe,return_all_record=False):
    """
    Print classification report and confusion matrix (cmtx) with categories in one-hot-encoder
    
    Input
        model (tensorflow.python.keras.engine.training.Model): tensorflow model
        X_test (numpy.ndarray): test set feature
        y_test (numpy.ndarray): test set label
        ohe (sklearn.preprocessing.OneHotEncoder):
        return_all_record (bool): 
        
    Return
        cmtx (pd.DataFrame): confusion matrix
    
    """
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    pred_c = ohe.inverse_transform(model.predict(X_test))
    actual_c = ohe.inverse_transform(y_test)
    report = classification_report(actual_c,pred_c,ohe.categories_[0].tolist())
    print(classification_report(actual_c,pred_c,ohe.categories_[0].tolist()))
    cf_matrix = confusion_matrix(actual_c,pred_c)
    cmtx = pd.DataFrame(confusion_matrix(actual_c,pred_c),
                        index=[f"actual: {i}"for i in ohe.categories_[0].tolist()], 
                        columns=[f"predict : {i}"for i in ohe.categories_[0].tolist()])
    if return_all_record == True:
        return pred_c, actual_c, report, cmtx
    return cmtx



def define_graph(layers,input_shape):
    """
    Define the graph that connect the layers, in sequential order
    
    Input
        layers (list<tensorflow.keras.layer>): keras layers, length of list must be larger than 2, immutable 
        input_shape (tuple): input shape
        
    Return
        graph (tensorflow.python.keras.engine.training.Model): the model, immutable 
    
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers[0](inputs,training=True)
    for layer in layers[1:-1]:
        x = layer(x,training=True)
    outputs = layers[-1](x,training=True)
    graph = tf.keras.Model(inputs=inputs, outputs=outputs)
    return graph

def cross_validation(model,dataset,batch_size=64,epochs=200):
    """
    Print classification report and confusion matrix (cmtx) with categories in one-hot-encoder
    
    Input
        model (tensorflow.python.keras.engine.training.Model): tensorflow model, model must be compiled
        dataset (source.dataset.DatasetObject): dataset object 
        batch_size (int): batch size
        epochs (int): number of epochs 
        
    Return
        model (tensorflow.python.keras.engine.training.Model): model, trained with the last fold of dataset 
        histories (list<float>): list of training histories
        scores (list<float>): list of evaluation result
    
    """
    histories,scores = [],[]
    num_fold = dataset.data.shape[-1]
    for i in range(num_fold):
        tf.keras.backend.clear_session()
        train,test = dataset.query([i])
        # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        history = model.fit(x=train[0],y=train[1],batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test[0],test[1]))
        score = model.evaluate(x=test[0],y=test[1], verbose=0)
        print(score[0],score[1])
        scores.append(score)
        histories.append(history)
    return model,histories,scores


def quick_history(ls_tensors):
    """
    Print and return list of tf.Tensor to numpy(float/int) object  
    """
    record = []
    for i in ls_tensors:
        record.append(i.numpy())
    print(record)
    return record


def quick_eval(model,train,test,target_train=None,target_test=None):
    """
    Print and return model loss and accuracy during custom training. 
    
    input:
        model (tensorflow.python.keras.engine.training.Model): model 
        train (tuple<numpy.ndarray>): tuple of array querred from DatasetObject 
        test (tuple<numpy.ndarray>): tuple of array querred from DatasetObject
        target_train (tuple<numpy.ndarray>):
        target_test (tuple<numpy.ndarray>):
    
    return:
        results (list<float>): model loss and accuracy
        result_names (list<string>): 
    
    """
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    record_s1 = model.evaluate(x=train[0],y=train[1], verbose=0)
    record_s2 = model.evaluate(x=test[0],y=test[1], verbose=0)
    if (target_train&target_train) != None:
        record_t1 = model.evaluate(x=target_train[0],y=target_train[1], verbose=0)
        record_t2 = model.evaluate(x=target_test[0],y=target_test[1], verbose=0)
        print(record_s1[-1]*100//1,record_s2[-1]*100//1,record_t1[-1]*100//1,record_t2[-1]*100//1)
        lb = ['source_train_loss','source_train_accuracy','source_test_loss','source_test_accuracy',
              'target_train_loss','target_train_accuracy','target_test_loss','target_test_accuracy']
        return [*record_s1,*record_s2,*record_t1,*record_t2],lb
    else:
        print(record_s1[-1]*100//1,record_s2[-1]*100//1)
        lb = ['train_loss','train_accuracy','test_loss','test_accuracy']
    
    return [*record_s1,*record_s2],lb
