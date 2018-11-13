import tensorflow as tf

class Model:
    
    def __init__(self):
        
        # TODO: Define network
        # ...

        # TODO: Loss and optimizer
        # ...

        # TODO: Start tensorflow session
        # ...

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
