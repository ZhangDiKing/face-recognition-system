# face-recognition-system
Really fun project, I collected the faces of my friends and made a small-scale design of face recognition system.
Follow the classic paper ["Face Recognition using Local Binary Patterns"](https://globaljournals.org/GJCST_Volume13/1-Face-Recognition-using-Local.pdf/), I used tradional ways to realize face recogniztion
      
## System Configuration
opencv-python + python 3.5 + sklearn

## Data collection
I used my web camera to collect the faces of my friends, the professor as well as myself and blured and rotated each 
sample to expend sample size.
I send my special thanks to my friends who help me build the dataset
      
## Feature extraction
Just as what the classic paper suggested, I only extracted the uniform features. I splite the face into several regions and gave different weight to them.
Especially 
> weight    = [[1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1],
                 [0,1,1,1,1,1,0],
                 [0,1,1,1,1,1,0],
                 [0,1,1,1,1,1,0],
                 [0,1,1,1,1,1,0]]


## PCA
For the split of region, the dimension of the feature for one face is very large, so I applied to reduced the 
dimension of feature into 220.

## SVM
As my sample size is very small, SVM seems to be a good choice. 
By grid search and compare kernels, I finally achieve 95% on my test dataset

## Recognition System
Simple GUI to plot the result. When someone appeared, the build-in face-detector of opencv detect his face and try to 
find out who in the training dataset look like him most. The right part is try to explain after several recognition 
process, the most similar face the one detected has.
