from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np


class Test():
    
    def __init__(self,sess=None):
        self.sess = sess
        return

    def compute_logits(self,x,sess):
        return sess.run(['dense_1/BiasAdd:0'],feed_dict={'input_audio:0':x})
