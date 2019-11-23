#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 00:14:38 2018

@author: root
"""

# Importing necessary libraries
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, Flatten
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from  keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import Callback
import sys
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from ddrop.layers import DropConnect
from imgaug import augmenters as iaa
from densenet import densenet121_model
#Declaring constants
FIG_WIDTH=20 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows
RESIZE_DIM= 64 # The images will be resized to 28x28 pixels

data_dir='/home/bengali.ai/data' #main directory
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))
paths_train_all=paths_train_a+paths_train_b+paths_train_c+paths_train_d+paths_train_e

#
paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
paths_test_all=paths_test_a+paths_test_b+paths_test_c+paths_test_d+paths_test_e+paths_test_f+paths_test_auga+paths_test_augc


path_label_train_a=os.path.join(data_dir,'training-a.csv')
path_label_train_b=os.path.join(data_dir,'training-b.csv')
path_label_train_e=os.path.join(data_dir,'training-e.csv')
path_label_train_c=os.path.join(data_dir,'training-c.csv')
path_label_train_d=os.path.join(data_dir,'training-d.csv')

def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key

def get_data(paths_img,path_label=None,resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X=[] # initialize empty list for resized images
    for i,path in enumerate(paths_img):
        img=cv2.imread(path,cv2.IMREAD_COLOR) # images loaded in color (BGR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # cnahging colorspace to RGB
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
#         X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        X.append(img) # expand image to 28x28x1 and append to the list
        # display progress
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    X=np.array(X) # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        df = pd.read_csv(path_label) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        y=to_categorical(y_label,10) # transfrom integer value to categorical variable
        return X, y
    
def imshow_group(X,y,y_pred=None,n_per_row=10,phase='processed'):
    '''helper function to visualize a group of images along with their categorical true labels (y) and prediction probabilities.
    Args:
        X: images
        y: categorical true labels
        y_pred: predicted class probabilities
        n_per_row: number of images per row to be plotted
        phase: If the images are plotted after resizing, pass 'processed' to phase argument. 
            It will plot the image and its true label. If the image is plotted after prediction 
            phase, pass predicted class probabilities to y_pred and 'prediction' to the phase argument. 
            It will plot the image, the true label, and it's top 3 predictions with highest probabilities.
    '''
    n_sample=len(X)
    img_dim=X.shape[1]
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,HEIGHT_PER_ROW*j))
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
#         img_sq=np.squeeze(img,axis=2)
#         plt.imshow(img_sq,cmap='gray')
        plt.imshow(img)
        if phase=='processed':
            plt.title(np.argmax(y[i]))
        if phase=='prediction':
            top_n=3 # top 3 predictions with highest probabilities
            ind_sorted=np.argsort(y_pred[i])[::-1]
            h=img_dim+int(img_dim/8)
            for k in range(top_n):
                string='pred: {} ({:.0f}%)\n'.format(ind_sorted[k],y_pred[i,ind_sorted[k]]*100)
                plt.text(img_dim/2, h, string, horizontalalignment='center',verticalalignment='center')
                h+=int(img_dim/8)
            if y is not None:
                plt.text(img_dim/2, -4, 'true label: {}'.format(np.argmax(y[i])), 
                         horizontalalignment='center',verticalalignment='center')
        plt.axis('off')
    plt.show()

def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)
    
#get data and save

X_train_a,y_train_a=get_data(paths_train_a,path_label_train_a,resize_dim=RESIZE_DIM)
X_train_b,y_train_b=get_data(paths_train_b,path_label_train_b,resize_dim=RESIZE_DIM)
X_train_c,y_train_c=get_data(paths_train_c,path_label_train_c,resize_dim=RESIZE_DIM)
X_train_d,y_train_d=get_data(paths_train_d,path_label_train_d,resize_dim=RESIZE_DIM)
X_train_e,y_train_e=get_data(paths_train_e,path_label_train_e,resize_dim=RESIZE_DIM)

X_train_all=np.concatenate((X_train_a,X_train_b,X_train_c,X_train_d,X_train_e),axis=0)
y_train_all=np.concatenate((y_train_a,y_train_b,y_train_c,y_train_d,y_train_e),axis=0)

X_test_a=get_data(paths_test_a,resize_dim=RESIZE_DIM)
X_test_b=get_data(paths_test_b,resize_dim=RESIZE_DIM)
X_test_c=get_data(paths_test_c,resize_dim=RESIZE_DIM)
X_test_d=get_data(paths_test_d,resize_dim=RESIZE_DIM)
X_test_e=get_data(paths_test_e,resize_dim=RESIZE_DIM)
X_test_f=get_data(paths_test_f,resize_dim=RESIZE_DIM)
X_test_auga=get_data(paths_test_auga,resize_dim=RESIZE_DIM)
X_test_augc=get_data(paths_test_augc,resize_dim=RESIZE_DIM)
#
X_test_all=np.concatenate((X_test_a,X_test_b,X_test_c,X_test_d,X_test_e,X_test_f,X_test_auga,X_test_augc))
#
np.save('x_train_all_64', X_train_all)
np.save('x_label_all_64', y_train_all)
np.save('x_test_all_128', X_test_all)
#

##load data
X_train_all = np.load('x_train_all_64.npy')
y_train_all = np.load('x_label_all_64.npy')

#define augmnetation
aug1 = iaa.GaussianBlur(sigma=(0, 2.0))
aug2 = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
aug3 = iaa.Multiply((0.8, 1.2), per_channel=0.2)
aug4 = iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-8, 8))

aug5 = iaa.CoarseDropout(p=0.2, size_percent = 0.15)
aug6 = iaa.ContrastNormalization((0.75, 1.5))
aug7 = iaa.Pepper(p=0.05)

def augment_img(img):
    
    i = np.random.randint(0,9)

    if i==0:
        img_adapteq = img

    elif i==1:
        img_adapteq = aug4.augment_image(img)
        img_adapteq = aug1.augment_image(img_adapteq)

    elif i==2:
        img_adapteq = aug2.augment_image(img)

    elif i==3:
        img_adapteq = aug3.augment_image(img)

    elif i==4:
        img_adapteq = aug4.augment_image(img)  
 
    elif i==5:
        img_adapteq = aug5.augment_image(img)  

    elif i==6:
        img_adapteq = aug6.augment_image(img)  
 
    elif i==7:
        img_adapteq = aug4.augment_image(img)
        img_adapteq = aug7.augment_image(img_adapteq)
        img_adapteq = aug1.augment_image(img_adapteq)

    elif i==8:
        img_adapteq = aug4.augment_image(img)
        img_adapteq = aug2.augment_image(img_adapteq)
    
    img_adapteq = img_adapteq.astype('float32')
    img_adapteq /= 255.

    return img_adapteq

class LearningRateDecay(Callback):
    '''Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    '''
    def __init__(self, decay, every_n=1, verbose=0):
        Callback.__init__(self)
        self.decay = decay
        self.every_n = every_n
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        if not (epoch and epoch % self.every_n == 0):
            return

        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        current_lr = K.get_value(self.model.optimizer.lr)
        new_lr = current_lr * self.decay
        if self.verbose > 0:
            print(' \nEpoch %05d: reducing learning rate' % (epoch))
            sys.stderr.write('new lr: %.5f\n' % new_lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        
def make_block(inp, k, depth):
    
    x1 = Conv2D(depth, (k, k), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda), dilation_rate = 1)(inp)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(depth, (k, k), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda), dilation_rate = 2)(inp)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(depth, (k, k), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda), dilation_rate = 4)(inp)
    x3 = BatchNormalization()(x3)
    return add([x1, x2, x3])



def cnn_with_multi_scale_high_level_feature_aggregation():
    
    inp = Input(shape=(height, width, depth))  

    x = Conv2D(16, (5, 5), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda))(inp)
    x = BatchNormalization()(x)
    x = Conv2D(16, (5, 5), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (5, 5), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) 

    x = make_block(x, 3, 64)
    x = make_block(x, 3, 64)
    x = make_block(x, 3, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
       
    x = Flatten()(x)
    x = DropConnect(Dense(128, activation='relu'), prob=0.2)(x)
    x = Dense(10, activation = 'softmax')(x)

    model = Model(inputs=inp, outputs=x) 
    return model

def multi_scale_all_level_feature_aggregation_cnn_model():
    
    inp = Input(shape=(64, 64, 3))  
 
    x = make_block(inp, 5, 16)
    x = make_block(x, 5, 16)
    x = make_block(x, 5, 16)
    x = MaxPooling2D(pool_size=(2, 2))(x) 

    x = make_block(x, 3, 32)
    x = make_block(x, 3, 32)
    x = make_block(x, 3, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    

    x = make_block(x, 3, 64)
    x = make_block(x, 3, 64)
    x = make_block(x, 3, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x) 

    x = make_block(x, 3, 128)
    x = make_block(x, 3, 128)
    x = make_block(x, 3, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation = 'softmax')(x)

    model = Model(inputs=inp, outputs=x) # To define a model, just specify its input and output layers
    return model


def all_cnn():
        
    inp = Input(shape=(height, width, depth)) 
    
    x = Conv2D(16, (5, 5), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda))(inp)
    x = BatchNormalization()(x)
    x = Conv2D(16, (5, 5), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (5, 5), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(32, (3, 3), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = Conv2D(64, (3, 3), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same',activation='relu',kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same',activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation = 'softmax')(x)

    model = Model(inputs=inp, outputs=x) # To define a model, just specify its input and output layers
    return model

def Densenet121():
    
    base_model = densenet121_model(img_rows=64, img_cols=64, color_type=3, num_classes=10, dropout_rate = 0.2)
    return base_model


def train_model(model, batch_size, epochs, x, y, n_fold, kf, model_name):

    i = 1

    for train_index, test_index in kf.split(x):
        
        x_train = x[train_index]; x_valid = x[test_index]
        y_train = y[train_index]; y_valid = y[test_index]

        x_valid = x_valid.astype('float32')/255.
        train_datagen = ImageDataGenerator(preprocessing_function=augment_img)
##
        train_datagen.fit(x_train)

        callbacks = [EarlyStopping(monitor='val_acc', patience=8, verbose=1, min_delta=1e-6),
             LearningRateDecay(0.5, every_n = 8, verbose=1),
             ModelCheckpoint(filepath= model_name + '_heavy_aug_fold_64' + str(i) + '.hdf5', verbose=1,monitor = 'val_acc',
                             save_best_only=True, save_weights_only=True, mode='auto')]
        
        model = model

        model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', 
                      metrics = ['accuracy'])

        model.fit_generator(train_datagen.flow(x_train, y_train, batch_size= batch_size),
                      epochs=epochs, verbose=1, validation_data = (x_valid, y_valid), callbacks = callbacks)

        model.load_weights(filepath= model_name + '_heavy_aug_fold_64' + str(i) + '.hdf5')



        i += 1

        if i <= n_fold:
            print('Now beginning training for {} fold {}\n\n'.format(model_name,i))
        else:
            print('Finished training {}\n\n!'.format(model_name))

models = [all_cnn(), multi_scale_all_level_feature_aggregation_cnn_model(), cnn_with_multi_scale_high_level_feature_aggregation(), Densenet121()]
model_name = ['all_cnn', 'multi_scale_all_level_feature_aggregation_cnn_model','cnn_with_multi_scale_high_level_feature_aggregation','Densenet121']

height, width, depth = 64, 64, 3 
num_classes = 10 
l2_lambda = 0.0001 # use 0.0001 as a L2-regularisation factor

#
from sklearn.model_selection import KFold
batch_size = 64
epochs = 30
n_fold = 5

##train start

for j in range(len(models)):  
    
    model = models[j]
    kf = KFold(n_splits=n_fold, shuffle=True)
    train_model(model, batch_size, epochs, X_train_all, 
                        y_train_all, n_fold, kf, model_name[j])
#    
##end training