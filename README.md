# AI Computer Vision for AR in CPP
A C++ project for training AI models on Yann LeCun's MNIST handwritten digits data set: http://yann.lecun.com/exdb/mnist/

TO RUN: Add train-images.idx3-ubyte and train-labels.idx1-ubyte binary files into the root project directory (they are too big to add to the repo). Comment out or in the different main() functions to test the prediction performance of the various models eg; KNN, Kmeans, CNN etc.

'Computer vision' in the sense that the hand written digits are imported into the project as binary data.

C++ pointers provide an efficient way of dealing with large data sets.

This repo is incomplete. The ultimate goal is to integrate this application with AR.
This repo is not procudtion quality code. It is adapted from the following YouTube tutorial: 
https://www.youtube.com/watch?v=E1K9SZCm0fQ&t=3601s
