# SILACT
The codes for Single-frame label-free cell tomography.

Please cite the following paper when using the code: 

https://arxiv.org/abs/2202.03627 

Ge, B., He, Y., Deng, M., Rahman, M.H., Wang, Y., Wu, Z., Wong, C.H.N., Chan, M.K., Ho, Y.P., Duan, L. and Yaqoob, Z., 2022. Single-frame label-free cell tomography at speed of more than 10,000 volumes per second. arXiv preprint

## contents
1. The codes for training the DNN
2. The codes for acquiring the ground truth 3D refractive index (RI) map with Beam Porpagation Method (BPM).
3. The codes for acquiring the Phase Approximants used as the inputs for training the DNN.
4. The codes for determining the absolute RI values and calculating the cellualr parameters, such as dry mass, volume, surface area and eccentricity...

## The codes for training the DNN
These codes are separated into two folders: 'tensorflow_scripts' and 'pytorch_scripts'. These are the same codes wirtten in two DNN training platforms. In each folder, we include the code for training the DNN-L, DNN-H and DNN-S.

## The codes for computing ground truth RI map

This part of codes are included in folder 'BPM_code_publish', partly adapted from Prof. Karmilov's paper in 2016 (https://ieeexplore.ieee.org/abstract/document/7384714/). The main file is 'BPM_HEK_cell_Rytov_6.m'. We do three things in this part:
1. Retrieve the phase maps from the interferograms of each illumination angles
2. Reconstruct an estimated 3D RI maps with Optical Diffraction Tomography algorithm under Rytov approximation as the initial guess for BPM algorithm.
3. Iterativly perfrom BPM algorithm to reconstruct the ground truth 3D RI map

## The codes for acquiring the input phase maps

This part of codes are included in folder 'input_phase_publish'. The main file is ’crude_phase_input.m‘.

## The codes for determining the absolute RI values and calculating the cellualr parameters

This part of codes are included in folder 'cell analysis'. 
1. Firstly we determine the absolute RI with linear fitting, which are written in 'Linear fitting.m'.
2. We track each cell in the video, and then calculate the cellular paprameters for analysis.
