import pandas as pd
import numpy as np
import tensorflow as tf



def identity(x):
    return x

def noised(x,const=1e-6):
    """
    Return gaussian noised sample.
    
    input:
        x (tensorflow.tensor): tensor  
        const (float): constant constrained the range of perturbation
        
    return:
        noised sample (tensorflow.tensor): 
    """
    rand = tf.random.normal(shape=tf.shape(x),dtype=x.dtype)
    rand = const*tf.nn.l2_normalize(rand,axis=range(1,len(rand.shape)))
    return tf.add(x,rand)

# Classifier loss
def loss_entropy(labels,logits):
    """
    Return Categorical crossentropy
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))

# Adversarial loss
def loss_adversarial(real,fake,loss='binary'):
    """
    Return a tuple of generator loss and discriminator loss. Currently available loss function: 'categorical', 'binary', 'wasserstein'
    """
    assert real.shape == fake.shape
    if loss == 'categorical':
        s_full = tf.concat((tf.ones((real.shape[0],1)),tf.zeros((real.shape[0],1))),axis=1)
        t_full = tf.concat((tf.zeros((fake.shape[0],1)),tf.ones((fake.shape[0],1))),axis=1)
        loss_func = tf.keras.losses.CategoricalCrossentropy(True)
        s_loss = loss_func(s_full,real)
        t_loss = loss_func(t_full,fake)
        d_loss = loss_func(s_full,fake)
    elif loss == 'binary':
        loss_func = tf.keras.losses.BinaryCrossentropy(True)
        s_loss = loss_func(tf.ones_like(real),real)
        t_loss = loss_func(tf.zeros_like(fake),fake)
        d_loss = loss_func(tf.ones_like(fake),fake)
    elif loss == 'wasserstein':
        loss_func = tf.keras.losses.BinaryCrossentropy(True)
        s_loss = loss_func(tf.ones_like(real),real)
        t_loss = loss_func(-1*tf.ones_like(fake),fake)
        d_loss = loss_func(tf.ones_like(fake),fake)
    else:
        raise Exception
    return d_loss,tf.reduce_mean([s_loss,t_loss])


# VADA(from 2017)
def perturb_image(x, p, classifier):    
    """
    Return perturb sample that would cause the greatest inconsistency in the model prediction, from
    """
    rand = tf.random.normal(shape=tf.shape(x))
    rand = 1e-6*tf.nn.l2_normalize(rand,axis=range(1,len(rand.shape)))
    rand = tf.cast(rand,x.dtype)
    with tf.GradientTape() as tape :
        tape.watch(rand)
        x_noise = tf.add(x,rand)
        p_noise = classifier(x_noise,training=False)
        loss = tf.reduce_max(tf.losses.KLDivergence('none')(p,p_noise))
    eps_adv = tape.gradient(loss,[rand])
    eps_adv = tf.nn.l2_normalize(eps_adv)[0]
    return x+eps_adv

def loss_vat(x, p, classifier):
    """
    Implementation of Virtual Adversarial Loss, from 
    """
    x_adv = perturb_image(x, p, classifier)
    p_adv = classifier(x_adv,training=False)
    return tf.keras.losses.CategoricalCrossentropy(False)(softmax(p),softmax(p_adv))


def loss_confident(x):
    """
    Implementation of Conditional CrossEntropy, from 
    """
    return tf.keras.losses.CategoricalCrossentropy(False)(softmax(x),softmax(x))
