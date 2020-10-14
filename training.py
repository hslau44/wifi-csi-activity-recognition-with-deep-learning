import tensorflow as tf
import numpy as np
import pandas as pd



    
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