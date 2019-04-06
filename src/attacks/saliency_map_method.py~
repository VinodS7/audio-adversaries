from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np



class SaliencyMapMethod():

    def __init__(self,model,nb_classes,theta=1.0,gamma=1.0,clip_min=-1.0,clip_max=1.0,y_target=None):
        self.model = model
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y_target = y_target
        self.nb_classes = nb_classes
        self.increase = (self.theta>0)
        return


    def build_attack(self,pcm):
        self.pcm = pcm
        
        y_target = tf.placeholder(tf.int32,name="target_labels")
        y_target_onehot = tf.cast(tf.one_hot(y_target,depth=self.nb_classes),tf.float32)
        self.nb_features = tf.placeholder(tf.int32,name="audio_len")
        self.search_domain= tf.placeholder(tf.float32,shape=[None])
        
        #zero_diagonal = np.ones([1,self.nb_features,self.nb_features])
        #zero_diagonal = tf.linalg.set_diag(zero_diagonal,0)

        logits = self.model.get_logits()
        preds = tf.nn.softmax(logits)
        preds_onehot = tf.one_hot(tf.argmax(preds,axis=1),depth=self.nb_classes)        
        
        if self.increase:
            search_domain = tf.reshape(tf.cast(self.pcm < self.clip_max, tf.float32), [-1, self.nb_features])
        else:
            search_domain = tf.reshape(tf.cast(self.pcm > self.clip_min, tf.float32), [-1, self.nb_features])

        
        list_derivatives = []
        

        for class_ind in range(self.nb_classes):
            derivatives = tf.gradients(logits[:,class_ind],self.pcm)
            list_derivatives.append(derivatives[0])

        grads = tf.reshape(tf.stack(list_derivatives),shape=[self.nb_classes,-1,self.nb_features])
        
        target_class = tf.reshape(tf.transpose(y_target_onehot,perm=[1,0]),shape=[self.nb_classes,-1,1])
        other_classes = tf.cast(tf.not_equal(target_class,1),tf.float32)


        grads_target = tf.reduce_sum(grads*target_class,axis=0)
        grads_other = tf.reduce_sum(grads*other_classes,axis=0)
        

        
        increase_coef = (4*int(self.increase) - 2) \
                * tf.cast(tf.equal(search_domain, 0),tf.float32)

        target_tmp = grads_target
        target_tmp -= increase_coef \
                * tf.reduce_max(tf.abs(grads_target),axis=1,keep_dims=True)
        
        target_sum = tf.reshape(target_tmp,shape=[-1,self.nb_features,1])\
                + tf.reshape(target_tmp,shape=[-1,1,self.nb_features])
       
        self.grads = target_sum
        other_tmp = grads_other
        other_tmp += increase_coef \
                * tf.reduce_max(tf.abs(grads_other),axis=1,keepdims=True)
        other_sum = tf.reshape(other_tmp,shape=[-1,self.nb_features,1]) \
                + tf.reshape(other_tmp,shape=[-1,1,self.nb_features])

        if self.increase:
            self.scores_mask = ((target_sum>0) & (other_sum <0))
        else:
            scores_mask = ((target<0) & (other_sum >0))

        #self.scores = tf.cast(self.scores_mask,tf.float32) \
        #        * (-target_sum * other_sum) * zero_diagonal

        #best = tf.argmax(tf.reshape(scores,shape=[-1,self.nb_features**2]),axis=1)
        #p1 = tf.mod(best,self.nb_features)
        #p2 = tf.floordiv(best,self.nb_features)

        #p1_one_hot = tf.one_hot(p1,depth=self.nb_features)
        #p2_one_hot = tf.one_hot(p2,depth=self.nb_features)

        #mod_not_done = tf.equal(tf.reduce_sum(y_in * preds_onehot,axis=1),0)
        #cond = mode_not_done & (tf.reduce_sum(domain_in,axis=1) >=2)

        #cond_float = tf.reshape(tf.cast(cond,tf_dtype),shape=[-1,1])
        #to_mod = (p1_one_hot + p2_one_hot)*cond_float

        #domain_out = search_domain - to_mod

        #to_mode_reshape = tf.reshape(to_mod,shape=([-1] + x_in.shape[1:].as_list()))
        #if increase:
        #    x_out = tf.minimum(clip_max,x_in+to_mod_reshape*theta)
        #else:
        #    x_out = tf.maximum(clip_min, x_in-to_mod_reshape*theta)

        #i_out = tf.add(i_in,1)

        #cond_out = tf.reduce_any(cond)

        return



    def attack(self,x,y_target,sess):

        #Initialize parameters for generating attacks
        
        if self.increase:
            domain_in = (x<self.clip_max).astype(float)
        else:
            domain_in = x>self.clip_min.astype(float)

        g = sess.run([self.grads],feed_dict={self.pcm:x,'target_labels:0':y_target,self.nb_features:x.shape[0],self.search_domain:domain_in})
        g = np.array(g)
        print(g.shape)
        return

