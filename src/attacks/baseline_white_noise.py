from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np

class BaselineWhiteNoise():
    def __init__ (self,model,sess=None,rms_ratio=10,clip_min=-1.0,clip_max=1.0):
        self.model = model
        self.sess = sess
        self.rms_ratio=rms_ratio
        self.clip_min = clip_min
        self.clip_max = clip_max
        return

    def build_attack(self,pcm):
        rms_in = tf.sqrt(tf.reduce_mean(tf.pow(pcm,2)))
        perturbation = tf.random_normal(tf.shape(pcm))
        rms_pert = tf.sqrt(tf.reduce_mean(tf.pow(perturbation,2)))
        self.adv_x = (rms_in/(rms_pert*self.rms_ratio))*perturbation + pcm
        return

    def attack(self,x,sess):

        a = sess.run([self.adv_x],feed_dict={'input_audio:0':x})
        a = np.squeeze(a)
        mae =np.mean((a-x)**2)
        x = a

        s = sess.run([self.model.get_probs()],feed_dict={'input_audio:0':x})
        s = np.squeeze(s)
        if (s.ndim == 1):
            print('Label number:',np.argmax(s))
            print('Label confidence:',np.max(s))
        else:
            s = np.max(s,axis=0)
            print('Label number:',np.argmax(s))
            print('Label confidence:',np.max(s))
            

        return a,mae,np.argmax(s),np.max(s)
