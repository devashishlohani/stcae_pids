# Spatio-Temporal Convolutional Autoencoders for Perimeter Intrusion Detection by video protection

This source code is for submission at workshop RRPR 2020.
We reproduce the results for Fall detection and extend 3D convolutional autoencoder for Intrusion detection task. 

**Code Usage:**

The code base is split into two main subsets

stcae_train.py -  For training different models

stcae_test.py - For testing different models 

**Training:**

To use this code, first run the training module. A model is then saved to Models/Dataset/....

Specify:

Dataset (Task) - dset = 'Thermal_Intrusion' or 'Thermal_Fall'

Model - Upsampling, Deconvolution or C3D

Number of epochs - 500 by default

and other parameters.

**Testing:**

To evaluate the model, run the test module. The results of testing will be saved to AEComparisons. 
Once training has completed, find the saved model under Models/Thermal/{model_name}. 
To evaluate the model, set the variable pre_load to the path to this model. 
Run stcae_test.py and find the results in AEComparisons. 
The Labels.csv file under each dataset provides the ground truth labels for start and end of fall frames.

**Generating Animation:**

To generate an animation:

run stcae_test.py, with animate option set to True. 

An animation (mp4 file) for each testing video will be saved to Animation folder.


**Requirements:**

Keras - 2.2.2  
Tensorflow - 1.10.0  
Python - 3.6.4

**Dataset Sharing:**  

Please contact authors of DeepFall paper for access to preprocessed data for Fall Detection.

For intrusion detection, we have a private dataset which unfortunately we cannot share.

Place the data in folder Datasets. See README.txt in Datasets for information on using the shared data.

**Results:**
All results are in AEComparisons folder. 