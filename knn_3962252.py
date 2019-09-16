# Bryan Medina
# CAP5610 - Machine Learning
# K Nearest Neighbors w/ Iris Dataset

####### Imports ########
import sys

import numpy as np
import pandas as pd
########################

# Command line arguments as a list are here
cmdargs = sys.argv

####### Classes ##########
class KNearestNeighbors():

    training_set = []
    K            = K
    
    def __init__(self, training_set, K):
        self.training_set = training_set
        self.K            = K

    def predict(self, test_set):
        '''
        'predict' uses training data to test whether it can make accurate predictions of the test set. This should work on either a single point or an array of points.
        '''
        # 1. Get the K nearest points from the data set.
        # 2. Figure out which class is voted on the most.
        # 3. Return that class. 
        return []
##########################

####### Functions ########
def Euclidean(a, b):
    '''
    Description
    -----------
    'Euclidean' calculates the L2 Norm between two vectors.

    Input
    -----
    'a': np.array. float array
    'b': np.array. float array

    Output
    ------
    scalar: np.Float
    '''   
    return np.linalg.norm(a-b)

def Cosine(a, b):
    '''
    Description
    -----------
    'Cosine' calculates the Cosine between two vectors.

    Input
    -----
    'a': np.array. float array
    'b': np.array. float array

    Output
    ------
    scalar: np.Float
    '''   
    return 1 - (np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def Panic(a, b):
    '''
    Description
    -----------
    'Panic' gives the user an error message.

    Input
    -----
    'a': np.array. float array
    'b': np.array. float array

    Output
    ------
    String
    '''   
    return "Enter either 'Euclidean' or 'Cosine' as a distance metric."

def parseArguments(cmdargs):
    '''
    Description
    -----------
    'parseArguments' parses the command line arguments, looking for the k value and also the distance metric.
    
    Input
    -----
    'cmdargs': list. List that contains the command line arguments
    
    Output
    ------
    k: integer. Contains the value k for the k-nearest neighbors.
    m: string. Contains the distance metric as a string.
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
k, m      = parseArguments(cmdargs) 
f         = (Euclidean if (m == "Euclidean") else ( Cosine if (m == "Cosine") else Panic ))
########################

####### K-Nearest Neighbors ##

# 1. Shuffle indices for the data set
fold_size  = K / KFolds
all_splits = np.zeros((KFolds, K))

# 2. Divide it into K groups and get indices for K groups
for i in range(0, KFolds):
    # We'll use the first 'fold_size' group as the test set. Rest is training. 
    all_splits[i] = np.random.permutation(np.arange(K))

# 3. Repeat KFolds times
for fold in range(KFolds):
    #    4. Reserve first subgroup for testing. Train and everything that isn't is in the test set
    test_set = all_splits[fold][0:(int(fold_size)-1)]
    training = all_splits[fold][int(fold_size):]

    assert len(test_set) >= 1
    assert len(training) >= 1
    
    #    5. Run K nearest neighbors on the training set
    KNN = KNearestNeighbors(training)
    
    #    6. Test with the test set.
    predictions = KNN.predict(test_set)
    
    #    7. Save the precision, recall.
    #       Compare the results to the actual values (you can get this from the data)
    
    #    8. Add to the confusion matrix
    
# 9. Report 

##################################

