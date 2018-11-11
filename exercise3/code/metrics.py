import math
import random
import numpy as np
import tensorflow as tf
#from sklearn.metrics import accuracy_score

def numpy_metrics(y_pred, y_true, n_classes, void_labels, dataset):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    """

    # Put y_pred and y_true under the same shape
    #y_pred = np.argmax(y_pred, axis=1)
    #y_true = y_true.flatten()
    #print("y pred shape = ", y_pred.shape)
    #print("y true shape = ", y_true.shape)
    
    #void_labels=0    
    # We use not_void in case the prediction falls in the void class of the groundtruth
    if dataset == 'camvid':
    	void_labels = void_labels
    	not_void = ~ np.any([y_true == void_labels], axis=0)
    else:
    	not_void = ~ np.any([y_true == void_labels], axis=0)

    #print("num non void", np.sum(not_void))
    #raw_input()
    #print("Not void shape = ", not_void)
    #print("Not_void ", not_void)
    #print("Calculating IoU")

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
    	#print(i)
        y_true_i = y_true == i 
        y_pred_i = y_pred == i
        #print("y_true_i", y_true_i)
        #print("y_true_i shape ", y_true_i.shape)
        
        #print("y_pred_i", y_pred_i)
        #print("y_pred_i shape ", y_pred_i.shape)


        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    denominator = np.sum(not_void)
    if denominator <= 0:
    	accuracy=0.0
    else:
    	accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy