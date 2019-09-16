# Bryan Medina
# CAP5610 - Machine Learning
# K Nearest Neighbors w/ Iris Dataset

####### Imports ########
import sys

import numpy as np
import pandas as pd
########################

# TODO: Need command line inputs... Get K and distance metric being used.
cmdargs = sys.argv

####### Functions ########
def Euclidean(a, b):
    '''
    Function definition here
    '''   
    return np.linalg.norm(a-b)

def Cosine(a, b):
    ''' 
    Function definition here
    '''
    return 1 - (np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def Panic(a, b):
    return "Enter either 'Euclidean' or 'Cosine' as a distance metric."

def parseArguments(cmdargs):
    '''
    Blah Blah parse arguments
    '''
    k = cmdargs[cmdargs.index('-k')+1]
    m = cmdargs[cmdargs.index('-m')+1]
    return k, m
##########################


####### Getting data ###
data      = pd.read_csv('/Users/bjm/Documents/School/fall2019/CAP5610/assignments/a1/data/iris.data', header=None) # CHANGE ME
KFolds    = 5 # Specify number of folds for k-fold cross validation
K         = len(data) # Number of examples of which we are going to split
confusion = np.zeros((3,3)) # Three classes, so confusion matrix is 3 X 3
k_neighs, m = parseArguments(cmdargs) 
f         = (Euclidean if (m == "Euclidean") else ( Cosine if (m == "Cosine") else Panic ))
########################

print(f(np.array([1,2]), np.array([1,1])))

# Getting every row
#for i in range(len(data)):
#    print(data.iloc[i])

