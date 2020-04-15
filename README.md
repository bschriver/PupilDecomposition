# PupilDecomposition

Decomposition_Fit.m : Function uses stochastic gradient descent to simultaneously update weights of time locked regressors (input : x) and learn kernels, to decompose observed pupil responses (input : y). File includes information regardings required model parameters as well as example parameter values.

Test_Run.m : File for example run to fit "TestDataSet.mat" and generate example of simple relevant plots of the learned kernels as well as the concatenated pupil trace.

TestDataSet.mat : File containing an example data set of 100 trials of pre-processed observed evoked pupil responses (y) and the initialized to 1 observed time-locked regressors (x) corresponding to 6 components of the decision-making process. 
