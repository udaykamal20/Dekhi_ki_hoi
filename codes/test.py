#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 00:14:48 2018

@author: root
"""

# Importing necessary libraries
import numpy as np
import os
import glob
import pandas as pd
from keras.layers import *
from train import cnn_with_multi_scale_high_level_feature_aggregation,multi_scale_all_level_feature_aggregation_cnn_model,all_cnn,Densenet121

#Declaring constants
FIG_WIDTH=20 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows
RESIZE_DIM= 64 # The images will be resized to 28x28 pixels

data_dir='/home/bengali.ai/data' #main directory

paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
paths_test_all=paths_test_a+paths_test_b+paths_test_c+paths_test_d+paths_test_e+paths_test_f+paths_test_auga+paths_test_augc

def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key

def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)
    
X_test_all_64 = np.load('x_test_all_64.npy')
X_test_all_64 = X_test_all_64.astype('float32')
X_test_all_64 /= np.max(X_test_all_64) 

models = [all_cnn(), multi_scale_all_level_feature_aggregation_cnn_model(), cnn_with_multi_scale_high_level_feature_aggregation(), Densenet121()]
model_name = ['all_cnn', 'multi_scale_all_level_feature_aggregation_cnn_model','cnn_with_multi_scale_high_level_feature_aggregation','Densenet121']

height, width, depth = 64, 64, 3 
num_classes = 10 

all_prob = np.zeros((len(models),5,len(X_test_all_64),10))

for j in range(len(models)):        
    for i in range(5):
        model = models[j]
        model.load_weights(filepath=model_name[j]+ '_heavy_aug_fold_64' + str(i+1) + '.hdf5')
        all_prob[j,i] = model.predict(X_test_all_64)

individ_model = np.average(all_prob, axis = 1)

indiv_max = np.average(individ_model, axis=0)
#indiv_max = np.max(individ_model, axis=0) 

labels=[np.argmax(pred) for pred in indiv_max]
keys=[get_key(path) for path in paths_test_all ]
create_submission(predictions=labels,keys=keys,path='submission_dense_3_4_5_avg_model.csv')