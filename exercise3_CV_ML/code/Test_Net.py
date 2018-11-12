from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import pprint
import scipy.misc
import sys

import cv2 as cv2
import cv as cv

from PIL import Image
from FCN import *
from loss import *

flags = tf.app.flags

flags.DEFINE_string("dataset_dir", "./data/", "Dataset directory default is data/")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("logs_path", "./logs/", "Directory name to save the log files")
flags.DEFINE_float("beta1", 0.90, "Momentum for adam")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam")
flags.DEFINE_integer("batch_size", 1, "The size of the sample batch")
flags.DEFINE_integer("img_height", 300, "Image Height")
flags.DEFINE_integer("img_width", 300, "Image Width")
flags.DEFINE_float("dropout", 1.0, "Dropout")
flags.DEFINE_float("steps_per_epoch", 500, "Steps per epoch")
flags.DEFINE_integer("max_steps", 100000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("load_Model", False, "Load Model Flag")
flags.DEFINE_string("model_path", "", "Load model from a  previous checkpoint")
flags.DEFINE_string("dataset", "CamVid", "Choose dataset, options [Camvid, ...]")
flags.DEFINE_integer("numberClasses", 12, "Number of classes to be predicted")
flags.DEFINE_string("version_net", "FCN_Seg", "Version of the net")
flags.DEFINE_boolean("Test", False, "Flag for testing")
flags.DEFINE_integer("lower_iter", 1000, "initial iteration to be tested - default 1000")
flags.DEFINE_integer("higher_iter", 40000, "final iteration to be tested - default 40000")
flags.DEFINE_string("IoU_filename", "testIoU.txt", "Nane to save the IoU and Iterattion [default is testIoU.txt]")
flags.DEFINE_integer("configuration", 4, "Set of configurations decoder [default is 4 - full decoder], other options are [1,2,3]")

FLAGS = flags.FLAGS

#  python  Test_Net.py --model_path=./checkpoints/ultraslim --version_net=FCN_Seg --dataset=CamVid --numberClasses=12 --lower_iter=30000 --higher_iter=40000

#  python  Test_Net.py --model_path=./checkpoints/ultraslimS --version_net=FCN_Seg --configuration=1

def main():

	# Loading Class
	FCN = FCN_SS()
	FCN.setup_inference(FLAGS.version_net, FLAGS.img_height, FLAGS.img_width, FLAGS.batch_size, FLAGS.Test, FLAGS.numberClasses, FLAGS.dataset, FLAGS.configuration)
	print("Building Test Graph OK!!!")

	saver = tf.train.Saver([var for var in tf.trainable_variables()])

	variables_to_restore = tf.trainable_variables()
	#variables_to_restore = slim.get_variables_to_restore()
	for v in variables_to_restore:
		print(v)

	imgs, label = FCN.loadTest_set(FLAGS)
	print("Test Set DONE!")
	print(imgs.shape)
	print(label.shape)

	with tf.Session() as sess:
		#sess.run(initializer)

		#Create text file
		name = FLAGS.model_path + '/' + FLAGS.IoU_filename
		f= open(name,"w+")

		number_of_iter = int((FLAGS.higher_iter - FLAGS.lower_iter) / 1000) + 1

		for i in range(number_of_iter):

			model = FLAGS.model_path + '/model-'
			if i==0:
				model_iter = model + str(FLAGS.lower_iter)
			else:
				increment = int(FLAGS.lower_iter) + (i*1000)
				model_iter = model + str(increment)
			print("model to be restored == ", model_iter)
			saver.restore(sess, model_iter)
			print("Model restored")
			Global_accuracy = 0.0
			Global_IoU = 0.0
			I_tot = 0.0
			U_tot = 0.0

			for test_idx in range(imgs.shape[0]):
				#timer counter
				start_time = time.time()

				results = FCN.inference(imgs[test_idx, :, :, :], sess)
				end_time = time.time() - start_time
				pred = results["Mask"]

				#print(pred.shape)
				segmentationMask = pred.argmax(axis=1)
				segmentationMask_flat = segmentationMask.reshape(FLAGS.img_height*FLAGS.img_width)
				segmentationMask = segmentationMask.reshape(FLAGS.img_height, FLAGS.img_width)
				#print(segmentationMask.shape)

				label_instance = label[test_idx,:, :].argmax(axis=1)
				label_instance = label_instance.reshape(FLAGS.img_height*FLAGS.img_width)
				#print(label_instance.shape)

				segmentationMask = pred.argmax(axis=1)
				label_Mask= label[test_idx,:, :].argmax(axis=1)
				list_void = [FLAGS.numberClasses -1]

				I, U, accuracy = numpy_metrics(segmentationMask, label_Mask, FLAGS.numberClasses -1, list_void, FLAGS.dataset)

				#print("I==", I)
				I_tot += I
				#print("U==", U)
				U_tot += U

				#Global_accuracy += accuracy
				Global_accuracy += accuracy

    				img=imgs[test_idx, :, :, 0:3].reshape(FLAGS.img_height, FLAGS.img_width, 3)
    				name = str('test/Img_in' + str(test_idx) + '.png')

				name = str('test/Seg_Out' + str(test_idx) + '.png')
				name2 = str('test/GT_Out' + str(test_idx) + '.png')

				segmentationMask = segmentationMask.reshape(FLAGS.img_height, FLAGS.img_width)

			print("Test Accuracy == %3f" % (Global_accuracy/imgs.shape[0]))

			I_tot=I_tot[:]
			U_tot=U_tot[:]


			print("Test IoU == ", (I_tot / U_tot))
			print("Test IoU == ", np.mean(I_tot / U_tot))
			Test_IoU = np.mean(I_tot / U_tot)
			iter_T = int(FLAGS.lower_iter) + (i*1000)
			f.write( str(iter_T) + ' ' + str(Test_IoU) + '\n' )


	f.close()

main()
