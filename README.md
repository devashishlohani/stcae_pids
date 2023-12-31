# Official implementation of "STCAE"

This is the implementation of our paper ["Spatio-Temporal Convolutional Autoencoders for Perimeter Intrusion Detection"](https://hal.science/hal-03145398v1/preview/Spatio-Temporal_Convolutional_Autoencoders_for_Perimeter_Intrusion_Detection.pdf) (RRPR 2021).
We propose 3D convolutional autoencoder for Fall detectoion and Intrusion detection task. 

**Installation**

1. Install conda from https://docs.conda.io/en/latest/miniconda.html depending on your OS.

2. Create conda environment from the given environment.yml file. 

   Go to root location of this project in your terminal and run the following command
   
   `conda env create -f environment.yml`
3. If there are errors, proceed with env.txt file.

   Run the following command

   `conda create --name stcae --file env.txt `
   
3. Activate conda environment

   `source activate stcae`

**Code Usage:**

The code base is split into two main subsets

stcae_train.py -  For training different models

stcae_test.py - For testing different models 

**Training:**

To use this code, first run the training module as:

`python3 stcae_train.py `

A model is then saved to Models/Dataset/....

Specify in stcae_train.py:

Dataset (Task) - dset = 'Thermal_Intrusion' or 'Thermal_Fall' or 'Thermal_Dummy'

Model - Upsampling, Deconvolution or C3D

Number of epochs - 500 by default

and other parameters.

**Testing:**

To evaluate the model, run the test module as:

`python3 stcae_test.py `
 
The results of testing will be saved to AEComparisons. 
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

**Illustrations:**

_**1) Fall Detection**_

![](fall_demo.gif)

**_2) Intrusion Detection_** 

![](intrusion_demo_1.gif)
![](intrusion_demo_2.gif)

## Bibtex
```
@inproceedings{lohani2021spatio,
  title={Spatio-temporal convolutional autoencoders for perimeter intrusion detection},
  author={Lohani, Devashish and Crispim-Junior, Carlos and Barth{\'e}lemy, Quentin and Bertrand, Sarah and Robinault, Lionel and Tougne, Laure},
  booktitle={International Workshop on Reproducible Research in Pattern Recognition},
  pages={47--65},
  year={2021},
  organization={Springer}
}
```
