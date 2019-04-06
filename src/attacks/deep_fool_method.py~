from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np

class DeepFool():

    def __init__(self,model,nb_candidate=5,overshoot=0.02,max_iter=200,clip_min=-1.0,clip_max=1.0):
        self.model=model
        self.nb_candidate=nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        return


    def build_attack(self,pcm):
        self.pcm = pcm
        self.logits = self.model.get_logits()
        self.batch_iter = tf.placeholder(tf.int32)
        self.nb_classes = self.logits.get_shape().as_list()[-1]
        assert self.nb_candidate <= self.nb_classes, \
                'nb candidate should not be greater than nb_classes'
        
        self.preds = self.logits
        #self.preds = tf.reshape(tf.nn.top_k(self.logits,k = self.nb_candidate)[0],
        #        [-1,self.nb_candidate])
        self.grads = tf.stack(jacobian_graph(self.preds[self.batch_iter,:],self.pcm,self.nb_classes), axis=1)
        return

    def attack(self,sess,sample,batch_size):
        adv_x = np.copy(sample)
        iteration = 0
        current = sess.run([self.model.get_probs()],feed_dict={self.pcm:sample})
        current = np.squeeze(current)
        if(current.ndim !=1):
            current = np.mean(current,axis=0)
        current = np.argmax(current)

        w = np.squeeze(np.zeros(sample.shape))
        r_tot = np.zeros(sample.shape)
        original = current
        while(np.any(current==original) and iteration <self.max_iter):

            gradients = np.zeros([batch_size,self.nb_classes,adv_x.shape[0]])
            for idx in range(batch_size):
                predictions,g = sess.run([self.logits,self.grads],feed_dict={self.pcm:adv_x,self.batch_iter:idx})      
                gradients[idx,:,:] = np.transpose(g)
            
            pert = np.inf

            w_k = np.zeros([batch_size,self.nb_classes,adv_x.shape[0]])
            pert_k = np.zeros([batch_size,self.nb_classes])

            for idx in range(batch_size):
                for k in range(1,self.nb_classes):
                    if(k==original):
                        continue
                    w_k[idx,k,:] = gradients[idx,k,:] - gradients[idx,original,:]
                    f_k = predictions[idx,k] - predictions[idx,original]
                    pert_k[idx,k] = (abs(f_k) + 0.00001) / np.linalg.norm(w_k[idx,k,:].flatten())
                
            pert_candidate = np.mean(pert_k,axis=0)
            if(np.max(pert_candidate) < pert):
                #print(np.argmax(pert_candidate))
                pert = np.max(pert_candidate)
                w_candidate = np.mean(w_k,axis=0)
                w = w_candidate[np.argmax(pert_candidate),:]
            r_i = pert* w/np.linalg.norm(w)
            r_tot += r_i
           
            #print(np.mean(r_tot))
            adv_x = np.clip((1+self.overshoot)*r_tot + sample,self.clip_min,self.clip_max)
            current = sess.run([self.model.get_probs()],feed_dict={self.pcm:adv_x})
            current = np.squeeze(current)
            
            if(current.ndim !=1):
                current = np.mean(current,axis=0)

            if(iteration %10 == 0):
                print('Current predictions:',np.argmax(current),'Current confidence:',np.max(current))
      
            current = np.argmax(current)
            iteration +=1
        

        return adv_x

def jacobian_graph(predictions,x,nb_classes):
    list_derivatives = []
    for class_ind in range(nb_classes):
        derivatives, = tf.gradients(predictions[class_ind],x)
        list_derivatives.append(derivatives)
    return list_derivatives
