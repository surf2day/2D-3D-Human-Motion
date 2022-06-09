# Human 2D to 3D uplift and motion prediction

This repo is the implementation of my CSIT999 Research Project. “Human Pose 2D to 3D Uplift and Prediction” a copy of which is included in this repo.

## Dependencies
- Tensorflow 
- Keras
- Numpy
- Matplotlib
- H5py
- random
- logging

This project utilises the code from HP-GAN project for the following. https://github.com/ebarsoum/hpgan

- Data pre-processing
- Sequence and pose image rendering
- Data sample generation (generator)

Refer the HP-GAN github for their original source code. HP-GAN programs have been modified for this project to achieve the following. The full code is supplied in this bundle inclusive of the modifications, given the extent of modifications supplying individual files for the modifications will be error prone.

- Reading and processing of Human3.6M cdf formatted files.
- Removal of dual subject activities in the NTU dataset
- Removal of incomplete or error sequences in the NTU dataset
- Added sequential sampling of sequences, original code supports random and same sample.
- Added Rendering of 2D skeleton sequences

## Code and Libraries

Following are the main programs files developed for this project

**train-mycode.py**  main driver code and model configuration, data preparation, training and testing. Note MPJPE and P-MPJPE code is adapted from MHFormer project https://github.com/Vegetebird/MHFormer  
**myPlotter.py**  custom plotting of training losses and visualisation of 3D skeletons and accuracy measures  
**NNetworks.py**	Contains the various RNN and Critic network configurations developed during the project. Also includes custom Keras layers for various functions such as, adding noise as z dimension of 2D poses.  In addition, includes generation functions for test rendering and 2D data normalisation functions.  
**split_h36_data.py**	 Human3.6M dataset pre processing, modified for cdf formats  
**split_ntu_data.py**  NTU-RGBD dataset pre-processing, modified for single action subjects and removal of incomplete sequences  
**exclusion-file.txt**  list of corrupted/incomplete NTU actions for pre-processing removal  
**braniac**  HP-GAN library, has been modified as above, recommend using version with this bundle  

## Datasets

The project trains and tests on both the NTU-RGBD and Human3.6M datasets and are available via the following websites.

NTU-RGBD https://rose1.ntu.edu.sg/dataset/actionRecognition/  
Human3.6M http://vision.imar.ro/human3.6m/description.php

## Dataset pre processing

Both datasets require pre processing as command lines below, the data generator reads the csv output files from these preprocessors.

python split_ntu_data.py -i \<path\>/nturgb+d_skeletons/ -o \<path to output\> -e exclusion-file.txt  
python split_h36_data.py -i \<path\>/Human-Ready -o \<path to output\>

## Training

While this code can be run from the command line it is not recommended, it was designed for use in the Spyder IDE (within anaconda), utilising the Cell based execution structure of Spyder. The train-mycode.py file contains all the various model implementations for 2D to 3D uplift and 2D to 3D uplift with prediction, noise injection in the RNN states and noise injection into the z joint dimension. It also contains cells after each model implementation for testing and generating results, including adding noise to the test data to simulate “read world” 2D joint and skeleton estimation from video. The cells are best run individually.

The first two commented lines of train-mycode.py contains the command line instructions for each dataset, add these to the Spyder IDE command line.   

As supplied the programs are configured for 2D to 3D uplift.  To change mode to 2D to 3D uplift and prediction requires several edits to the code as follows.

**train-mycode.py** change lines 
1.  748 output_sequence_length = 20
2.  Uncomment lines 632, 633
3.  Comment out lines 634, 635

**NNetworks.py**
1.	Comment out line 898
2.	Uncomment line 899
3.	Uncomment line 907

Depending on the cell being run, it will be necessary to uncomment the models fit call to start the training.

Reverse these changes to return to 2D to 3D uplift.

The layout and structure are the result of a research project rather than a software engineering project, so feel free to improve it.

**Monitoring** – during training the various training losses and validation losses are written to tensorboard.  Tensorboard and plots are written to the results/output folder.

At the end of each epoch summaries are logged and sample sequences written to results/2D_representation. All testing is written to the results/test folder.

No specific GPU specific coding was implemented it will train on CPU however, GPU is recommended as it significantly reduces training cycles.

