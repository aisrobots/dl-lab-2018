from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
from loss import *
from metrics import *
from nets_definition import * 

class FCN_SS(object):
	def __init__(self):
		pass

	def build_train_graph(self):
		opt=self.opt

		print("ALL Ok GIRL")
		print(opt.dataset_dir) 
		
		#class weights Sitting
		class_weight = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
		class_weight = tf.reshape(tf.convert_to_tensor(class_weight, dtype=tf.float32), [12])

		self.train_image_batch = tf.placeholder(tf.float32, [None, opt.img_height, opt.img_width, 3])	
		self.train_label_batch = tf.placeholder(tf.float32, [None, opt.img_height*opt.img_width, opt.numberClasses])
		
		self.tgt_image = self.train_image_batch
		self.tgt_label = self.train_label_batch
		self.tgt_label = self.train_label_batch
		#train_image_batch, train_label_batch = self.camvid_batches()
		self.N_classes = opt.numberClasses
		self.keep_prob = opt.dropout
		self.batch_size = opt.batch_size
		self.width = opt.img_width
		self.height = opt.img_height
		self.Training = True
		self.random=False
		self.total_steps = opt.max_steps
		self.dataset= opt.dataset
		self.version_net = opt.version_net
		self.configuration = opt.configuration

		with tf.name_scope("Net_prediction"):
			if self.version_net == 'FCN_Seg':
				print("Computing FCN_Seg encoder and decoder")
				segMap = FCN_Seg(self, is_training=self.Training)
			print("Output FCN_Seg")
			print(segMap)
			
		with tf.name_scope("Output_Metrics"):
			segmentationMask = tf.argmax(segMap,axis=1)
			# 	print("Segmentation mask")
			# 	print(segmentationMask)
			Smask = tf.placeholder(tf.float32, [None, opt.img_height*opt.img_width])
			Smask = tf.reshape(segmentationMask, (self.batch_size, opt.img_height*opt.img_width))
			segmentationLabel = tf.argmax(self.train_label_batch,axis=2)
			# 	print("Segmentation Label")
			# 	print(segmentationLabel)
			print(Smask)
			print(segmentationLabel)
			equality = tf.equal(Smask, segmentationLabel)
			accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
			
			
		with tf.name_scope("compute_loss"):
			#Compute Softmax loss of D-Net
			Reshaped_labels = tf.reshape(self.train_label_batch, (-1, self.N_classes))
						
			if opt.version_net =='FCN_Seg':
				TotalLoss = loss(segMap, Reshaped_labels, self.N_classes,class_weight)
				
			print(TotalLoss)

		with tf.name_scope("Training"):
			print("==================== Training =================================")
			train_vars = [var for var in tf.trainable_variables()] 
			
			global_step = tf.train.get_or_create_global_step()
			self.global_step = global_step
			self.incr_global_step = tf.assign(self.global_step, self.global_step+1)
			
			############# DECAY ############################
			print("Exponential Decay .......")
			Decay=0.90
			# Decay the learning rate exponentially based on the number of steps.
			lr = tf.train.exponential_decay(opt.learning_rate, global_step,	(self.total_steps/2), Decay, staircase=True)
			
			############ OPTIMIZER #########################		
			#select Adam as optimizer 
			optimizer = tf.train.AdamOptimizer(lr, opt.beta1)
			
			self.grads_and_vars = optimizer.compute_gradients(TotalLoss, var_list=train_vars)
			self.Training = optimizer.apply_gradients(self.grads_and_vars)
			self.learning_rate = lr
			

		#Collect tensors that are useful later (tf summary)
		self.predMask = segMap
		self.total_loss = TotalLoss
		self.steps_per_epoch = opt.steps_per_epoch
		self.accuracy = accuracy
		
		self.version_net = opt.version_net
		
		# merge all summaries into a single "operation" which we can execute in a session 
		#merged = tf.summary.merge_all()
		self.logs_path = opt.logs_path
		
		print("DONE BUILDING GRAPH!")

	def collect_summaries(self):
		opt = self.opt
		
		#losses
		tf.summary.scalar("total_loss", self.total_loss)
		tf.summary.scalar("accuracy", self.accuracy)
		
	def Load_TrainDataset(self):
			print("Loading training Set")
			opt=self.opt
			#build train and val sets 
			#Train SET
			self.img_width=opt.img_width
			self.img_height=opt.img_height
			train_path = opt.dataset_dir + 'Train_data_' + opt.dataset + '.npy'
			#print(train_path)
			train_data = np.load(train_path)
			train_data = train_data.reshape((train_data.shape[0], opt.img_height, opt.img_width, 3))	
			print(train_data.shape)
			
			train_label_path = opt.dataset_dir + 'Train_label_' + opt.dataset + '.npy' 
			train_label =  np.load(train_label_path)
			train_label = train_label.reshape((train_data.shape[0], opt.img_height*opt.img_width, 1))
			# shape is (376, 50176, 12)
			print(train_label.shape)

			return train_data, train_label
	
	def Create_batches(self, train_data, train_label):
		
		train_image_batch=np.zeros((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
		#train_label_batch=np.zeros((self.batch_size, self.img_height*self.img_width, self.N_classes), dtype=np.float32)
		train_label_batch=np.zeros((self.batch_size, self.img_height*self.img_width, self.N_classes), dtype=np.float32)
		
		self.random=True


		if(self.counter>=(train_data.shape[0] -1)) and (self.random==False):
			self.counter=0
			self.random=True
			print("RANDOM TRAINING :)")
		else:
			self.counter+=1

		#print(train_label_batch.shape)
		for i in range(self.batch_size):
			#train sample
			if (self.random==False): 
				train_image_batch[i,:,:,:] = train_data[self.counter+i, :, :, :]
				#Test Sample 
				train_label_batch[i,:,:] = self.unfould(train_label[self.counter+i, :, :])
			else:
				#print("RANDOM TRAINING :)")
				index = random.randint(0, train_data.shape[0] -1)
				train_image_batch[i,:,:,:] = train_data[index, :, :, :]
				#Test Sample 
				train_label_batch[i,:,:] = self.unfould(train_label[index, :, :])
				#train_label_batch[i,:,:] = (train_label[index, :, :])


		#train_label_batch = tf.reshape(train_label_batch, (-1, self.batch_size*self.img_height*self.img_width))
		return train_image_batch, train_label_batch
	
	def unfould(self, image):
		#print(image.shape)
		image = np.reshape(image, (1, self.height, self.width, 1))
		label_training = np.zeros([1,  self.height, self.width, self.N_classes])
		#for batch in range(0, self.batch_size):
		for row in range(0, self.height):
			for column in range(0, self.width):
				#print("label", int(image[0,row,column,0]))
				#if(int(image[0,row,column,0]) >= 0): # and (int(image[0,row,column,0]) != 11):
				label_training[0,row,column,int(image[0,row,column,0])] = 1
				# else:
				# 	label_training[0,row,column,int(image[0,row,column,0])] = 1
		label_training = np.reshape(label_training, (1, self.height*self.width, self.N_classes))
		#print(label_training.shape)			
		return label_training

	def train(self, opt):
		TotalLOSS = 0.0
		self.opt = opt
		self.batch_size = opt.batch_size
		self.counter = 0
		self.build_train_graph()
		print("Building Graph OK!!!")
		self.collect_summaries()

		# Merge all summary inforation.
  		summary = tf.summary.merge_all()

		print("=================Collection Variables ok!!!!==========================")
		

		#self.setup_inference(opt.version_net, opt.img_height, opt.img_width, opt.batch_size_val)
		#print("Building Validation Graph OK!!!")
		### Load Validation Set
		print("dataset type ", self.dataset)
		imgs, label = self.loadValidation_set(opt)
		print("Validation Set DONE!")
		print(imgs.shape)
		print(label.shape)
		
		Accuracy_validation = 0.0
		Iou_validation = 0.0
		
		with tf.name_scope("parameter_count"):
			parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
		
		self.saver = tf.train.Saver([var for var in tf.trainable_variables()], max_to_keep=100)
		self.saver2 = tf.train.Saver([var for var in tf.trainable_variables()] + [self.global_step], max_to_keep=100)
		

		sv = tf.train.Supervisor(logdir=opt.logs_path, save_summaries_secs=0, saver=None)
		print("passed")

		with sv.managed_session() as sess:

			#train_writer = tf.summary.FileWriter(self.logs_path + '/train', sess.graph)
			# Create a writer for the summary data.
			#summary_writer = tf.summary.FileWriter(self.logs_path, sess.graph)

			print('Trainable variables: ')
			for var in tf.trainable_variables():
				print(var.name)
			print("parameter_count =", sess.run(parameter_count))
			#restore the latest model - 
			if opt.continue_train:
				print("Resuming training ")
				checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
				#print(checkpoint)
				self.saver2.restore(sess, checkpoint)


			if opt.load_Model:
				print("Loading Model..... ")
				print(opt.model_path)
				#global_step_zero = tf.get_variable('global_step2', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
				self.saver.restore(sess, opt.model_path)

			##Load Training dataset
			train_data, train_label = self.Load_TrainDataset()

			
			print("=================== Starting Iterations =========================")
			global_start_time = time.time()
			for step in range(1, opt.max_steps):
				start_time = time.time()
				fetches = {"Training": self.Training, "global_step": self.global_step, "incr_global_step": self.incr_global_step}
				
				#train_image_batch, train_label_batch = self.Create_batches(train_data, train_label)
				train_image_batch, train_label_batch = self.Create_batches(train_data, train_label)

				
				#train_image_batch_ = np.reshape(train_image_batch, (opt.batch_size, opt.img_height*opt.img_width*3))
				#train_label_batch_ = np.reshape(train_label_batch, (opt.batch_size, opt.img_height*opt.img_width*12))

				if step % opt.summary_freq == 0:
					fetches["loss"] = self.total_loss
					
				fetches["learning_rate"] = self.learning_rate
				fetches["loss_iter"] = self.total_loss
				#fetches["merged"] = self.merged
				fetches["summary"] = sv.summary_op
				
				
				#self.Training=True
				#self.Set_batch_size(sess, opt.batch_size)
				#print("Input values")
				#print(train_image_batch.shape)
				#print(train_label_batch.shape)
				results = sess.run(fetches, feed_dict={self.train_image_batch:train_image_batch, self.train_label_batch:train_label_batch})
				gs = results["global_step"]
				LR = results["learning_rate"]
				#summ = results["merged"]

				#print(results["loss"]) 
				TotalLOSS = TotalLOSS + results["loss_iter"] 
				
				# write log
				#writer.add_summary(summary, step)
				#train_writer.add_summary(summ, step)

				train_epoch = math.ceil(gs / (opt.steps_per_epoch))
				#Print loss and time
				if step % opt.summary_freq == 0:
					sv.summary_writer.add_summary(results["summary"],step)
					train_step = gs - (train_epoch -1) * (opt.steps_per_epoch)
					print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %3f" % (train_epoch, train_step, opt.steps_per_epoch, time.time() - start_time, results["loss"]))

				Global_accuracy = 0.0
				Global_accuracy2 = 0.0

				Global_IoU = 0.0
				I_tot = 0.0
				U_tot = 0.0

				NumberIterations_train_to_test=((opt.steps_per_epoch))
				
				#Saving model 
				if ((step % (NumberIterations_train_to_test*2)) == 0): # and (Accuracy_validation < Global_IoU):
					Accuracy_validation = Global_IoU
					#Save all Variables
					self.save(sess, opt.checkpoint_dir, gs)

				#train_writer.close()
			print('Global training time == ', time.time() - global_start_time)


	def save(self, sess, checkpoint_dir, step):
		model_name = 'model'
		print("Saving checkpoint to %s..." % checkpoint_dir)
		if step == 'latest':
			self.saver2.save(sess, os.path.join(checkpoint_dir, model_name + '.latest'))
		else:
			#save all Variables
			self.saver2.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

				
	def setup_inference(self, version_net, img_height, img_width, batch_size, Test, N_classes, dataset, configuration):
		self.img_height = img_height
		self.img_width = img_width
		self.batch_size = batch_size
		self.version_net = version_net
		self.Test = Test
		self.N_classes = N_classes
		self.dataset = dataset
		self.configuration = configuration
		#Call build test_graph
		self.build_test_graph()
		


	def build_test_graph(self):
		input_img = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, 3], name='input_image')

		self.tgt_image = input_img
		self.keep_prob = 1.0
		self.batch_size = self.batch_size
		self.width = self.img_width
		self.height = self.img_height
		self.class_balance = True
		

		with tf.name_scope("segmentation_prediction"):
			if self.version_net == 'FCN_Seg':
				segMap = FCN_Seg(self, is_training=True)	
			print("Output FCN_Seg")
			print(segMap)

		self.inputs = input_img
		self.predMask = segMap
		

	
	def inference(self, inputs, sess):
		fetches = {}
		inputs = inputs.reshape((1, self.height, self.width, 3))	
		fetches['Mask'] = self.predMask
		results = sess.run(fetches, feed_dict={self.inputs:inputs})
		return results

		

	def loadTest_set(self, opt):
		# #Test SET 
		test_path = opt.dataset_dir + 'Test_data_' + opt.dataset + '.npy'
		#test_path = opt.dataset_dir + 'RealTest_data_Kitti.npy'
		
		print(test_path)
		test_data = np.load(test_path)
		test_data = test_data.reshape((test_data.shape[0], self.height, self.width, 3))	
		print(test_data.shape)
		
		test_label_path = opt.dataset_dir + 'Test_label_' + opt.dataset + '.npy'
		print(test_label_path) 
		test_label =  np.load(test_label_path)
		test_label = test_label.reshape((test_label.shape[0], self.height*self.width, 1))

		if(self.Test==True):
			test_label = np.zeros((test_data.shape[0], self.height*self.width, 1))

		#shape is (233, 50176, 12)
		print(test_label.shape)
		test_label_batch=np.zeros((test_label.shape[0], self.height*self.width, self.N_classes), dtype=np.float32)
		for i in range(test_label.shape[0]):
			test_label_batch[i,:,:] = self.unfould(test_label[i,:,:])

		return test_data, test_label_batch

	def loadValidation_set(self, opt):
		#Test SET 
		#test_path = opt.dataset_dir + 'Test_data_' + 'Camvid2' + '.npy'
		test_path = opt.dataset_dir + 'Test_data_' + opt.dataset + '.npy'
		
		print(test_path)
		test_data = np.load(test_path)
		test_data = test_data.reshape((test_data.shape[0], self.height, self.width, 3))	
		print(test_data.shape)
		
		#test_label_path = opt.dataset_dir + 'Test_label_' + 'Camvid2' + '.npy'
		test_label_path = opt.dataset_dir + 'Test_label_' + opt.dataset + '.npy'
		
		print(test_label_path) 
		test_label =  np.load(test_label_path)
		test_label = test_label.reshape((test_label.shape[0], self.height*self.width, 1))

		print(test_label.shape)
		## Not performing one-hot-encoding here anymore
		
		# return test_data, test_label_batch	
		return test_data, test_label
	   
