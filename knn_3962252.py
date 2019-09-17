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

class_idx = 0 # variable used to keep assign an index to eat class (used later for the confusion matrix

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
        predicted = 'Iris-versicolor'
        actual    = 'Iris-virginica'
        
        # 1. Get the K nearest points from the data set.
        # 2. Figure out which class is voted on the most.
        # 3. Return that class. 
        return classes[predicted], classes[actual]
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
    print("Pick either 'Euclidean' or 'Cosine' as a distance metric.")
    sys.exit()

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


####### Getting data and initializing variables ###
data      = pd.read_csv('/Users/bjm/Documents/School/fall2019/CAP5610/assignments/a1/data/iris.data', header=None) # CHANGE ME
classes = {} # Dictionary will store class name with index

# Code to get the 
for cl in data.iloc[:][4]:
    if not (cl in classes.keys()):
        classes[cl] = class_idx
        class_idx = class_idx + 1

KFolds    = 5 # Specify number of folds for k-fold cross validation
K         = len(data) # Number of examples of which we are going to split
confusion = np.zeros((3,3)) # Three classes, so confusion matrix is 3 X 3
k, m      = parseArguments(cmdargs)

f         = (Euclidean if (m == "Euclidean") else Cosine)
assert (f == Euclidean or f == Cosine)
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

    for test in test_set:
        #    6. Test with the test set.
        predicted, actual = KNN.predict(test)

        #    7. Save the precision, recall and add to confusion matrix
        #       Compare the results to the actual values (you can get this from the data)
        confusion[predicted][actual] += 1

    
# 9. Report 

##################################

