from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(prediction, labels, num_classes, label_balance):
    """Calculate the loss from the prediction and the labels.

    Args:
      prediction: tensor, float - [batch_size*width*height, num_classes].
       
      labels: Labels tensor, int32 - [batch_size*width*height, num_classes]
          The ground truth of your data.
      label_balance: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        #prediction = tf.reshape(prediction, (-1, num_classes))
        #epsilon = tf.constant(value=1e-4)
        epsilon = tf.constant(value=2e-4)
        #labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
        
        labels = tf.to_float(labels)
        #print("label size")
        #print(labels.shape)
        softmax = tf.nn.softmax(prediction) + epsilon
        #print("softmax size")
        #print(softmax.shape)

        if label_balance is not None:
        	#print("Weight balance!!!!")
        	cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), label_balance), reduction_indices=[1])
        else:
          cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1])
        #print("cross_entropy size")
        #print(cross_entropy.shape)


        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        #print("loss size")
        #print(loss.shape)

    return loss
