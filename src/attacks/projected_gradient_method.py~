from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
from attacks import fast_gradient_method as FGM_CLASS

class ProjectedGradientDescent():
    
    def __init__(self,model,rms_ratio=10,default_rand_init=True):
        self.model = model
        self.rms_ratio=rms_ratio
        self.default_rand_init = default_rand_init
    

    def build_attack(self,pcm):
        logits = self.model.get_logits()
        self.label = tf.placeholder(tf.int64,name='labels')
        label = tf.reshape(self.label,shape=[-1])
        fgsm = FGM_CLASS.FastGradientMethod(rms_ratio=self.rms_ratio,clip_min=-1.0,clip_max=1.0)
        self.adv_x = fgsm.fgm(pcm,logits,label)
        return

    def attack(self,x,label,nb_iter,sess):
        
        for iter in range(nb_iter):
            a = sess.run([self.adv_x],feed_dict={'input_audio:0':x,self.label:label})
            a = np.squeeze(a)
            mae=np.mean((a-x)**2)
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

