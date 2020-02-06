import math 
import numpy as np
import sys


#Creating the functions

#Function to calculate entropy of a random variable
def entropy(array,rv):
    tropy = 0
    label_names,freq = np.unique(array[rv],return_counts = True)
    prob_rv = freq/freq.sum()
    for x in prob_rv:
        if (x==0):
            tropy += 0
        else:
            tropy += -x*math.log(x,2)
    return(tropy)


#Function to calculate joint entropy of two random variables
def joint_entropy(array,rv1,rv2):
    tropy = 0
    label_names,freq = np.unique(array[[rv1,rv2]],return_counts = True)
    prob_rv1_rv2 = freq/freq.sum()
    for x in prob_rv1_rv2:
        if (x==0):
            tropy += 0
        else:
            tropy += -x*math.log(x,2)
    return(tropy)
    
    
#Function to calculate mutual information
def mutInf(array,y,f):
    H_Y = entropy(array,y)
    H_f = entropy(array,f)
    H_Y_f = joint_entropy(array,y,f)
    mi = H_Y + H_f - H_Y_f
    return(mi)

class Node:
    def __init__(self, val, left = None, right = None,leftarm = None,rightarm = None):
        self.val = val
        self.leftnode = left
        self.rightnode = right
        self.leftarm = leftarm
        self.rightam = rightarm
        
        
# Function to learn the decision tree 
def dt_train(data,features,depth):
    labels,freq = np.unique(data[y],return_counts = True)
    label_freq = dict(zip(labels,freq))
    guess = max(label_freq.keys(),key = lambda x: label_freq[x])
    if (depth == 0):
        return(Node(guess))
    elif (len(np.unique(data[y]))==1):
        return(Node(guess))
    elif (len(features)==0):
        return(Node(guess))
    else:
        score = {}
        for f in features:
            score[f] = mutInf(data,y,f)
        f = max(score.keys(),key = lambda x: score[x])
        
        left_subset = data[data[f]==np.unique(data[f])[0]]
        labels_left,freq_left = np.unique(left_subset[y],return_counts = True)
        label_freq_left = dict(zip(labels_left,freq_left))
        leftarm = np.unique(data[f])[0]
        
        right_subset = data[data[f]==np.unique(data[f])[1]]
        labels_right,freq_right = np.unique(right_subset[y],return_counts = True)
        label_freq_right = dict(zip(labels_right,freq_right))
        rightarm = np.unique(data[f])[1]
        
        features.remove(f)
        featurescopy = features.copy()
        depth -= 1
        
        print('| '*(max_depth -depth),f,' = ',leftarm,': ',label_freq_left)
        left = dt_train(left_subset,features,depth)
        print('| '*(max_depth -depth),f,' = ',rightarm,': ',label_freq_right)
        right = dt_train(right_subset,featurescopy,depth)
        return Node(f,left,right,leftarm = leftarm, rightarm = rightarm)



# Function to predict label of an instance using the learned decision tree
def dt_predict(row,node):
    if (node.leftnode == None and node.rightnode == None):
        return(node.val)
    elif (row[node.val] == node.leftarm):
        return(dt_predict(row,node.leftnode))
    else:
        return(dt_predict(row,node.rightnode))

#Function to calculate error rate
def errorRate(labels,predicted_labels):
    error = 0
    for i in range(len(labels)):
        if (labels[i] != predicted_labels[i]):
            error += 1
    return(error/len(labels))

#unpacking the command line arguments
train_input, test_input, max_depth, train_out, test_out, metrics_out = sys.argv[1:]
            

d_train = np.genfromtxt(fname=train_input, delimiter="\t",dtype = str,autostrip = True)
data_train = np.core.records.fromarrays(d_train[1:].transpose(),names=tuple(d_train[0]))

d_test = np.genfromtxt(fname=test_input, delimiter="\t",dtype = str,autostrip = True)
data_test = np.core.records.fromarrays(d_test[1:].transpose(),names=tuple(d_test[0]))

max_depth = int(max_depth)


features_train = list(data_train.dtype.names[:-1])
features_test = list(data_test.dtype.names[:-1])
y = data_train.dtype.names[-1] #column name of the labels column
labels_train = data_train[y]
labels_test = data_test[y]

#Learning the decision tree using training data
labels,freq = np.unique(data_train[y],return_counts = True)
print(dict(zip(labels,freq)))
tree = dt_train(data_train,features_train,depth = max_depth) 

#Predicting training data labels
prediction_train = []
for row in data_train:
    prediction_train.append(dt_predict(row,tree))

#Predicting test data labels
prediction_test = []
for row in data_test:
    prediction_test.append(dt_predict(row,tree))

#Calculating training and testing error
error_train = errorRate(labels_train,prediction_train)
error_test = errorRate(labels_test,prediction_test)

#Output
prediction_train = [x+'\n' for x in prediction_train]
prediction_test = [x+'\n' for x in prediction_test]
open(train_out,'w').writelines(prediction_train)
open(test_out,'w').writelines(prediction_test)
open(metrics_out,'w').write('error(train):'+str(error_train)+'\n'+'error(test):'+str(error_test))




