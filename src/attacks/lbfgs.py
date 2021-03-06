from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np

class LBFGS():

    def __init__(self,model,binary_search_steps=5,max_iterations=1000,initial_const=1e-1,clip_min=-1.0,clip_max=1.0):
        self.model=model
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

        return

    def build_attack(self,pcm):
        self.pcm = pcm
        self.logits = self.model.get_logits()
        self.targeted_label = tf.placeholder(tf.int64,shape=[None],name="label_names")
        self.shape = tuple(list(self.pcm.get_shape().as_list()))
        self.original_audio = tf.placeholder(tf.float32,shape=pcm.shape,name='original_audio')
        self.const = tf.placeholder(tf.float32,name = 'const')
        self.score = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targeted_label,logits=self.logits)
        self.l2dist = tf.reduce_sum(tf.square(self.pcm - self.original_audio))
        self.loss = tf.reduce_sum(self.score * self.const) + self.l2dist
        self.grad,=tf.gradients(self.loss,self.pcm)

        return

    def attack(self,sess,x_val,targets):

        def lbfgs_objective(adv_x,self,targets,oaudio,CONST):
            loss,grad = sess.run([self.loss,self.grad],feed_dict={self.pcm:adv_x.reshape(oaudio.shape),self.targeted_label:targets,self.original_audio:oaudio,self.const:CONST})
            return loss,grad.flatten().astype(float)
        
        from scipy.optimize import fmin_l_bfgs_b

        self.repeat = self.binary_search_steps >= 10
        oaudio = np.clip(x_val,self.clip_min,self.clip_max)
        CONST = self.initial_const

        lower_bound = 0.
        upper_bound = 1e10

        clip_min = self.clip_min * np.ones(oaudio.shape[:])
        clip_max = self.clip_max * np.ones(oaudio.shape[:])
        clip_bound = list(zip(clip_min.flatten(),clip_max.flatten()))

        o_bestl2 = [1e10]
        o_bestattack = np.copy(oaudio)

        for outer_step in range(self.binary_search_steps):

            if(self.repeat and outer_step == self.binary_search_steps):
                CONST = upper_bound

            adv_x,_,_ = fmin_l_bfgs_b(lbfgs_objective,oaudio.flatten().astype(float),args=(self,targets,oaudio,CONST),bounds=clip_bound,maxiter=self.max_iterations,iprint=0)
            
            adv_x = adv_x.reshape(oaudio.shape)
            
            
            assert np.amax(adv_x) <= self.clip_max and \
                    np.amin(adv_x) >= self.clip_min,\
                    'fmin_l_bfgs_b returns are invalid'

            preds = sess.run([self.model.get_probs()],feed_dict={self.pcm:adv_x})
            preds = np.squeeze(preds)
            if(preds.ndim==1):
                print(np.max(preds),np.argmax(preds))
                preds = np.argmax(preds)
            else:
                print(np.max(preds,axis=1),np.argmax(preds,axis=1))
                preds = np.mean(preds,axis=0)
                print(np.argmax(preds),np.max(preds))
                preds = np.argmax(preds)
                

            l2 = 0.

            l2 = np.mean(np.square(adv_x - oaudio))
            

            target = int(np.mean(targets)) 
            if(l2<o_bestl2 and preds == target):
                o_bestl2 = l2
                o_bestattack = adv_x

            if(preds == target):
                upper_bound = min(upper_bound, CONST)
                if upper_bound<1e9:
                    CONST = (lower_bound + upper_bound)/2.0
            else:
                lower_bound = max(lower_bound,CONST)
                if upper_bound <1e9:
                    CONST = (lower_bound + upper_bound)/2.0
                else:
                    CONST *= 10
                
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2<1e9]))
                 
        o_bestl2 =  np.array(o_bestl2)
        return o_bestattack
