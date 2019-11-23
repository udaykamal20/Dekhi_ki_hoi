# Kaggle_Numta-Bengali-Handwritten-Digit-Recognition
This repository contains the codes and submission package for Team 'dekhi_ki_hoi', that stood 3rd position in the Kaggle Competition, Numta: Bengali Handwritten Digit Recognition The task was to build a classification model for Bengali handwritten digits that is robust even under severe noisy condition.

![alt text](https://github.com/udday2014/Kaggle_Numta-Bengali-Handwritten-Digit-Recognition/blob/master/standings.png)

The details of the competition and the datasets can be found in the following link: 
https://www.kaggle.com/c/numta/overview


# Pipeline Description

The whole approach is multi-cnn based ensemble method. multiple cnn models have been trained on the training data using 5 fold-modified snapshot ensemble technique. A custom data augmentation technique was built to train all the models on the augmented data which were produced on line during the training. The final result was the average of the prediction of the individual models. 

# Result Generation

* all the provided data folders should be placed in the 'bengali.ai/data/' directory.
* all the provided folders and files in the codes directory of our submission are required to be placed in the 'bengali.ai/codes/' directory.
* train.py under codes directory is to be run.
* all the generated weights will be saved in codes directory with proper name.
* test.py under codes directory is to be run, corresponding csv file will be generated.

# Preprocessing Details

Total  types of augmentations were done. They are:

1) GaussianBlur
2) AdditiveGaussianNoise
3) Channel wise random scaling
4) Affine scaling + translation + rotation + shear
5) Coarse dropout
6) ContrastNormalization
7) Additive Pepper noise

The parameters for all the augmentations were selected randomly within a pre-specified range.
All the preprocessing was done using the imgaug, an open source image augmentation library. the pipeline was as follows:

-- for every image:
--generate a random number(i) between 0-9

  if i==0:
      no augmentation
  elif i==1:
      type 4 + type 1 augmentation
  elif i==2:
      type 2 augmentation
  elif i==3:
      type 3 augmentation    
  elif i==4:
      type 4 + type 5 augmentation
  elif i==5:
      type 5 augmentation 
  elif i==6:
      type 6 augmentation  
  elif i==7:
      type 4 + type 7 + type 1 augmentation
  elif i==8:
      type 4 + type 2 augmentation

--finally normalize the augmented/not augmented(based on the value of i) image.


# Models description
    
Total 4 types of models have been used. 3 custom built models and 1 pre-trained(on imagenet) densenet121 model.

Type_1: all_cnn: 4 consecutive blocks following a globalaveragepooling2D and dense(10) layer as the final output with softmax activation. each block contained three successive conv-relu-bn layer following a maxpool layer. filter size was incremented from 16-32-64-128 in the sucessive blocks. first block had 5x5 conv kernel and all other blocks had conv kernel size of 3x3

Type_2: cnn_with_multi_scale_high_level_feature_aggregation: a special, custom block has been designed for this competition which was used in type 2 and typ3 model. this block had three dilated_conv-relu layer with increasing dilation rate(1,2,4). all the three layers took same input to produce multi-scale features and finally a sum block was used to fuse these multi-scale features.
        in this model, first two blocks were used from the previous model to produce high level features, then this newly designed block was used to generate multi-scale high-level features. after that the flattened features were fed into two dense layer(128, and 10). for the dense(128) layer, a droconnect layer (where some of the dense connections between two dense layers are severed randomly) was used to better learn the important feature inter-connection, insted of dropout layer, where feature importance was main concern. 

Type_3: multi_scale_all_level_feature_aggregation_cnn_model: here the newly built blocks were used from the begining. total 4 blocks, each followed by a maxpool, and finally a globalaveragepooling and a dense layer producing output probability was used. 

Type_4: Densenet121: a pre-trained and built in model was used. the pre-trained weight was generated using imagenet dataset. (link of open source weight: https://drive.google.com/file/d/0Byy2AcGyEVxfSTA4SHJVOHNuTXc/view?pli=1 )            
