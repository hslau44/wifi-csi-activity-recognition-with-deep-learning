import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_addons as tfa
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16
import dataset,utils,training
from datetime import datetime

### define parameters 
INPUT_SHAPE = (1000,90,1) 
ACTIVATION = None

### function define
lrelu = tf.keras.layers.LeakyReLU
softmax = tf.keras.layers.Softmax
normalization = tfa.layers.InstanceNormalization
identity = tf.keras.layers.Lambda(lambda x:x)

### Gradient reverse layer (Yanin 2015)
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverseLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradReverseLayer, self).__init__()

    def call(self, x):
        return grad_reverse(x)
    

### define model architectures

### Shallow Convolutional Neural network
def build_model(label_size,normalize=False,seperate=False,latent_shape=None):
    """
    Shallow Convolutional Neural network model, change architecture here
    
    Input
        label_size (tuple): label size
        normalize (bool): add normalization layers 
        seperate (bool): if True return encoder,classifier and discriminator
        latent_shape (None/tuple): return an classifier and discriminator, given seperate == True and latent_shape != None
        
    """
    ### element define
    disc_input  = tf.keras.layers.Input((1000,90,1))
    disc_conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides = (5,5))
    disc_conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides = (3,3))
    disc_conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), strides = (2,2))
    disc_flaten = tf.keras.layers.Flatten()
    disc_lab_d1 = tf.keras.layers.Dense(128,activation=lrelu())
    disc_lab_d2 = tf.keras.layers.Dense(label_size)
    disc_dmn_d1 = tf.keras.layers.Dense(128,activation=lrelu())
    disc_dmn_d2 = tf.keras.layers.Dense(1) # change here for categorical crossentropy 
    ### activation
    activation_1 = tf.keras.layers.ReLU()
    activation_2 = tf.keras.layers.ReLU()
    activation_3 = tf.keras.layers.Activation('tanh')
    ### pooling 
    pooling_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,1))
    pooling_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,1))
    pooling_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,1))
    dropout_l = tf.keras.layers.Dropout(0.1)
    dropout_d = tf.keras.layers.Dropout(0.1)
    ### normalization
    disc_norm_c1 = normalization()
    disc_norm_c2 = normalization()
    
    ### graph define
    if normalize == True:
        x = disc_conv_1(disc_input)
        x = disc_norm_c1(x)
        x = activation_1(x)
        x = pooling_1(x)
        x = disc_conv_2(x)
        x = disc_norm_c2(x)
        x = activation_2(x)
        x = pooling_2(x)
        x = disc_conv_3(x)
        x = activation_3(x)
        x = pooling_3(x)
    else:   
        x = disc_conv_1(disc_input)
        #x = disc_norm_c1(x)
        x = activation_1(x)
        x = pooling_1(x)
        x = disc_conv_2(x)
        #x = disc_norm_c2(x)
        x = activation_2(x)
        x = pooling_2(x)
        x = disc_conv_3(x)
        x = activation_3(x)
        x = pooling_3(x)
    p = disc_flaten(x)
    if seperate == True:
        ### encoder
        encoder = tf.keras.models.Model(inputs=disc_input, outputs=p)
        ### shape
        if latent_shape != None:
            lab_input = tf.keras.layers.Input(latent_shape)
            dmn_input = tf.keras.layers.Input(latent_shape)
        else:
            lab_input = tf.keras.layers.Input(p.shape[1:])
            dmn_input = tf.keras.layers.Input(p.shape[1:])
        ### label
        x1 = disc_lab_d1(lab_input)
        x1 = dropout_l(x1)
        o1 = disc_lab_d2(x1)
        lab = tf.keras.models.Model(inputs=lab_input, outputs=o1)
        ### domain 
        x2 = disc_dmn_d1(dmn_input)
        x2 = dropout_d(x2)
        o2 = disc_dmn_d2(x2)
        dmn = tf.keras.models.Model(inputs=dmn_input, outputs=o2)
        return encoder,lab,dmn
    else:
        x1 = disc_lab_d1(p)
        x1 = dropout_l(x1)
        o1 = disc_lab_d2(x1)
        x2 = disc_dmn_d1(p)
        x2 = dropout_d(x2)
        o2 = disc_dmn_d2(x2)
        model = tf.keras.models.Model(inputs=disc_input, outputs=[o1,o2])
        return model 

    
### VGG16 (Simonyan 2014)    
def create_VGG(num=8,top=True):
    base_model = VGG16(include_top=False,input_shape=(1000, 90, 3))
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=(1000, 90, 3))
    x = base_model(inputs, training=False)
    p = tf.keras.layers.GlobalAveragePooling2D()(x)
    if top == True:
        p = tf.keras.layers.Dense(128,activation='relu')(p)
        outputs = tf.keras.layers.Dense(num)(p)
        model = tf.keras.Model(inputs, outputs)
    else:
        model = tf.keras.Model(inputs, p)
    return model



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



if __name__ == '__main__':
    
    ### CHANGE HERE
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    folds = [6] # train test split fold/s from DatasetObject, fold k1 and k2 will be selected as testing set for folds = [k1,k2]

    
    ### build model 
    encoder,classifier,_ = build_model(8,normalize=True,seperate=True)
    model = define_graph([encoder,classifier],INPUT_SHAPE)
    
    
    ### Dataset, we pick Self dataset EXP1 as sample 
    folderpath1 = "./data/exp_1"  # CHANGE THIS IF THE PATH CHANGED
    df_exp1 = dataset.import_dataframe(folderpath1) # CHANGE THIS IF THE PATH CHANGED
    df_exp1 = df_exp1.drop([f"theta_{i}" for i in range(1,91)], axis=1) # pre-processing 
    dataset_exp1 = dataset.DatasetObject(df_exp1,cnn=True,stacking=False) # DatasetObject
    ohe_y = dataset_exp1.one_hot(1) # one-hot encoding (activities)
    ohe_z = dataset_exp1.one_hot(2) # one-hot encoding (user)
    
    
    ### training setup 
    tf.random.set_seed(1234)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=200) 
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) # complie model
    train,test = dataset_exp1.query(folds) # quering data
    history = model.fit(x=train[0],y=train[1],batch_size=128,epochs=100,verbose=1,validation_data=(test[0],test[1]),callbacks=[callback])
    cmtx = utils.evaluation(model,test[0],test[1],ohe=ohe_y)
    
    
    ### Save model and record result 
    encoder.save(f'./saved_model/encoder_{current_time}.h5')
    classifier.save(f'./saved_model/classifier_{current_time}.h5')
    model.save(f'./saved_model/model_{current_time}.h5')
    print("model saved")
    
    pd.DataFrame(history.history).to_csv(f"./record/{recordfilename}_history_{hist}_Fold7.csv",sep=' ')
    cmtx.to_csv(f"./record/cmtx_model_{current_time}.csv")
    print("record saved")
    

 

   