import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    gray =  2 * gray.astype('float32') - 1 
    return gray 


def action_to_id(a):
    """ 
    this method discretizes actions
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.8]): return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0
