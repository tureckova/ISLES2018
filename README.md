# ISLES2018
Keras implementation of U-shaped Deep Convolution Neural Network for image segmentation.

The code was written to be trained using the ISLES2018 data set for brain lesion segmentation, but it can be easily modified to be used in other 3D applications. The code was derived form the repository 3D U-Net Convolution Neural Network with Keras created by ellisdg: https://github.com/ellisdg/3DUnetCNN.

For more information about models and trining please refer to paper: The code was written to be trained using the BRATS data set for brain tumors, but it can be easily modified to be used in other 3D applications.

Short code files explanation:

correct_bias.py
- data preparation
- needs ANT4 bias correction installed https://github.com/ANTsX/ANTs/releases

find_mean_std.py
- find dataset statistic 

train.py
- training file

train_cross_validation.py
- training file for 5-fold cross validation

trainYellowFin.py
- training file using yellowfish optimizer (training goes faster, but gives slightly worse results)

plot_loss.py
- plots png file with val and train loss into output folder

predictNiiFile.py
- predicts outputs and saves them to nii files
- if there are GT, calculates the dice coeficients

predictNiiFile_cross_validation.py
- predicts test outputs and saves them to nii files iterating throught different models to cover all training data
- if there are GT, calculates the dice coeficients
