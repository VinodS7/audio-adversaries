from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np




class CarliniWagnerAttack():

    def __init__(self,model,confidence=0,learning_rate=5e-3,binary_search_steps=5,max_iterations=1000,abort_early=True,initial_const=1e-2,clip_min=-1.0,clip_max=1.0,targeted =False):
        self.model=model
        self.confidence = confidence
        self.learning_rate=learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations=max_iterations
        self.abort_early=abort_early
        self.initial_const=initial_const
        self.clip_min = clip_min
        self.clip_max=clip_max
        self.targeted=targeted
        return

    


    def build_attack(self,pcm):
        self.pcm = pcm
        self.original_audio = tf.placeholder(tf.float32,shape=[None])
        self.modifier_init = tf.placeholder(tf.int32,name='mod_init')
        self.modifier = tf.Variable(tf.zeros(shape=self.modifier_init),validate_shape=False)
        self.labels = tf.placeholder(tf.int32,name='labels')
        self.const = tf.placeholder(tf.float32,name='const')
        
        self.newpcm = (tf.tanh(self.modifier+self.pcm)+1)/2.0
        self.newpcm = self.newpcm*(self.clip_max-self.clip_min)+self.clip_min
        
        saver = self.model.build_graph(self.newpcm)
        self.output = self.model.get_logits()

        self.other = (tf.tanh(self.original_audio)+1)/2.0*(self.clip_max-self.clip_min)+self.clip_min

        self.l2dist = tf.reduce_mean(tf.square(self.newpcm-self.other))
       
        self.labels = tf.cast(self.labels,tf.float32)
        real = tf.reduce_sum((self.labels)*self.output,1)
        other = tf.reduce_max((1-self.labels)*self.output-self.labels*10000,1)
        if self.targeted:
            loss1 = tf.maximum(0.,other-real+self.confidence)
        else:
            loss1 = tf.maximum(0.,real-other+self.confidence)

        self.loss2= tf.reduce_sum(self.l2dist)
        self.loss1= tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss, var_list=[self.modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)           
        return saver
    
    
    def attack(self,sess,input_audio,labels,batch_size):
        
        lab = int(np.mean(labels))
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.targeted:
                    x[y] -= self.confidence
                else:
                    x[y] += self.confidence
                x = np.argmax(x)
            if self.targeted:
                return x == y
            else:
                return x != y

        batch_size=int(batch_size)
        
        original_audio = np.clip(input_audio,self.clip_min,self.clip_max)
        input_audio = original_audio
        original_audio = np.arctanh(original_audio*.999999)
         
        b = np.zeros((batch_size, 41))
        b[np.arange(batch_size), labels] = 1
        labels = b
        
        self.repeat = self.binary_search_steps>=10
        
        
        lower_bound = 0.
        CONST = self.initial_const
        upper_bound = 1.*1e10
        
        o_bestl2 = [1e10]
        o_bestscore = [-1]
        o_bestattack = np.copy(original_audio)


            
        for outer_step in range(self.binary_search_steps):
            sess.run([self.init],feed_dict={self.modifier_init:input_audio.shape})
            audio = input_audio
            bestl2 = 1e10
            bestscore = -1 
            if self.repeat and outer_step == self.binary_search_steps-1:
                CONST = upper_bound


            prev = 1e6
            for iteration in range(self.max_iterations):
                _,l,l2,score,npcm = sess.run([self.train,self.loss,self.l2dist,self.output,self.newpcm],feed_dict={self.pcm:audio,self.original_audio:original_audio,self.labels:labels,self.const:CONST})
                    
                l = np.squeeze(l)
                l2 = np.squeeze(l2)
                score = np.squeeze(score)
                score = np.exp(score)/sum(np.exp(score))
                if(score.ndim !=1):
                    score = np.mean(score,axis=0)
                if(iteration % ((self.max_iterations//10) or 1)==0):
                    print('Loss:',l)
                    print('L2 distance:',l2)
                    print('New label:',np.argmax(score))
                    print('New label confidence:',np.max(score))
                    print(np.mean((npcm-audio)**2))

                
                if(self.abort_early and \
                    iteration%((self.max_iterations//10) or 1)==0):
                    if(l>prev*0.9999):
                        break

                    prev = l
                if(l2<bestl2 and compare(score,lab)):
                    bestl2 = l2
                    bestscore = np.argmax(score)

                if(l2<o_bestl2 and compare(score,lab)):
                    o_bestl2 = l2
                    o_bestscore = np.argmax(score)
                    o_bestattack = npcm

                    s = sess.run([self.output],feed_dict={self.pcm:audio})
                    s = np.squeeze(s)
                    s = np.exp(s)/sum(np.exp(s))
                    if(s.ndim==1):
                        print(np.argmax(s),np.max(s),l2)
                    else:
                        print(np.argmax(s,axis=1),np.max(s,axis=1),l2)



            if(compare(int(bestscore),lab) and bestscore !=-1):
                upper_bound = min(upper_bound,CONST)
                if(upper_bound<1e9):
                    CONST = (lower_bound + upper_bound)/2.0
                else:
                    lower_bound = max(lower_bound,CONST)
                if(upper_bound<1e9):
                    CONST = (lower_bound+upper_bound)/2.0
                else:
                    CONST*=10

                o_bestl2 = np.array(o_bestl2)
                
        return o_bestattack
