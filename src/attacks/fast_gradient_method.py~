
from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np



class FastGradientMethod():

    def __init__ (self,rms_ratio=0.001,clip_min=-1.0,clip_max=1.0,ord=np.inf,targeted=False):
        self.rms_ratio = rms_ratio
        self.clip_min = clip_min
        self.clipmax = clip_max
        self.ord = ord
        self.targeted=targeted

        return

    def fgm(self,x,logits,label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits)
        grad, = tf.gradients(loss,x)
        
        rms_x = tf.sqrt(tf.reduce_mean(tf.pow(x,2)))
        optimal_perturbation = self.optimize_linear(grad,rms_x)
        if self.targeted:
            optimal_perturbation=-optimal_perturbation

        adv_x = x +optimal_perturbation

        return adv_x

    def optimize_linear(self,grad,rms_x):
        
        red_ind = list(range(1,len(grad.get_shape())))
        avoid_zero_div = 1e-12

        if(self.ord == np.inf):
            optimal_perturbation = tf.sign(grad)
            optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        else:
            print('Not implemented boss.')

        rms_pert = tf.sqrt(tf.reduce_mean(tf.pow(optimal_perturbation,2)))
        return (rms_x/(rms_pert*self.rms_ratio))*optimal_perturbation

