# Bryan Medina
# CAP5610 - Machine Learning
# K Nearest Neighbors w/ Iris Dataset

####### Imports ########
from sklearn.metrics import confusion_matrix

import sys
import time

import numpy as np
import pandas as pd
########################

# This plots the confusion matrix at the end of the program
show_matrix = True

if(show_matrix):
    import matplotlib.pyplot as plt

# Command line arguments as a list are here
cmdargs = sys.argv

class_idx = 0 # variable used to keep assign an index to eat class (used later for the confusion matrix

####### Classes ##########
class KNearestNeighbors():

    training_set = []
    
    def __init__(self, training_set, K, f, labels):
        
        self.training_set = training_set # Training set (list of indices that contain training data in dataset)
        self.K            = int(K)       # Number of neighbors
        self.f            = f            # Distance metric
        self.labels       = labels       # All class labels

    def predict(self, test_set):
        '''
        'predict' uses training data to test whether it can make accurate predictions of the test set. This should work on either a single point or an array of points.
        '''
        # Remember votes for each class
        votes     = {}
        for label in self.labels:
            votes[label] = 0

        # Get the actual value of the test examplea
        actual    = test_set[4]
        closets   = np.zeros((self.K,1))
        distances = []

        # for each index in the list of training examples
        for example in self.training_set:
            train_coord = np.array(data.iloc[int(example)][0:4])
            test_coord  = np.array(test_set[0:4])

            distance = self.f(train_coord, test_coord)
            distances.append((distance, int(example)))

            
        # 1. Get the K nearest points from the data set.
        topK = getMaxK(distances, self.K)
        
        # 2. Figure out which class is voted on the most.
        for closest in topK:
            votes[data.iloc[closest[1]][4]] += 1
        
        # 3. Return that class.
        v = list(votes.values())
        k = list(votes.keys())

        predicted = k[v.index(max(v))]

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

def getMaxK(numbers, k):
    '''
    Description
    -----------
    'getMaxK' returns indices of the first k max elements.

    Input
    -----
    'numbers': list. List of tuples, where first element is the distance, and second in the index of the training example.
               Assumed that the list is unordered.
    'k'      : int.  number of max values we want.

    Output
    ------
    scalar: np.Float
    '''

    # We want to sort the list by the first element of each tuple (which is the distance from the training example to the test example)
    # We'll use the function below as the key for sorting.
    def takeFirst(element):
        return element[0]

    numbers.sort(key = takeFirst)

    # Return the first k elements
    return numbers[:k]
    

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


####### Getting data / initializing variables ###
data      = pd.read_csv('/home/bjm/Documents/School/fall2019/CAP5610/assignments/a1/data/iris.data', header=None) # CHANGE ME
classes   = {} # Dictionary will store class name with index

# Code to get the 
for cl in data.iloc[:][4]:
    if not (cl in classes.keys()):
        classes[cl] = class_idx
        class_idx = class_idx + 1

KFolds    = 5 # Specify number of folds for k-fold cross validation
K         = len(data) # Number of examples of which we are going to split
confusion = np.zeros((class_idx,class_idx)) # Three classes, so confusion matrix is 3 X 3
k, m      = parseArguments(cmdargs)
assert (int(k) > 0), ("%s is not a positive integer. Please enter an integer greater than 0." % (k))

f         = (Euclidean if (m == "Euclidean") else (Cosine if (m == "Cosine") else Panic)) # Getting specified distance metric
assert (f == Euclidean or f == Cosine) , ("'%s' is not a distance metric. Please enter either 'Euclidean' or 'Cosine'." % (m))
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
    
    # 4. Reserve first subgroup for testing. Train and everything that isn't is in the test set
    test_set = all_splits[fold][0:int(fold_size)]
    training = all_splits[fold][int(fold_size):]

    assert len(test_set) >= 1
    assert len(training) >= 1
    
    # 5. Run K nearest neighbors on the training set
    KNN = KNearestNeighbors(training, k, f, classes.keys())

    for test in test_set:
        # 6. Test with the test set.
        predicted, actual = KNN.predict(data.iloc[int(test)])
        
        # 7. Save the precision, recall and add to confusion matrix
        #    Compare the results to the actual values (you can get this from the data)
        confusion[predicted][actual] += 1

# 8. Report
accuracy = (np.sum([confusion[i][i] for i in range(len(confusion))]) / K) * 100

if(show_matrix):
    plt.matshow(confusion)
    plt.title("Confusion Matrix for %s-NN, %s Distance Metric, %.2f %% Accuracy" % (k, m, accuracy))
    plt.colorbar()
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()
else:
    print("Confusion Matrix:")
    print(confusion)
    print("\nAccuracy: %.2f %%" % (accuracy))
##################################

