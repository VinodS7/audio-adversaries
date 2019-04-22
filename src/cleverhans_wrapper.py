from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
from cleverhans.model import Model

class CleverHansModel():

    def __init__(self,checkpoint,sample_rate,generator,mel_filt):
        
        self.checkpoint = checkpoint 
        self.sample_rate = sample_rate
        self.generator = generator
        self.mel_filt = mel_filt
        return

    def _compute_features(self,pcm):
    
        stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=512,
                                           fft_length=1024,pad_end=True)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        mel_spectrograms = tf.tensordot(tf.pow(spectrograms,2), self.mel_filt, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
         self.mel_filt.shape[-1:]))

        max_val = tf.reduce_max(mel_spectrograms,axis=None)


        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = 10*((tf.log(mel_spectrograms + 1e-6)-tf.log(max_val+1e-6))/tf.log(tf.constant(10,dtype=tf.float32)))
        log_mel_spectrograms = tf.reshape(log_mel_spectrograms,shape=[-1,128,64,1])
        log_mels = self.generator.standardize(log_mel_spectrograms)        
        return log_mels

    
    def build_graph(self,pcm):
        log_mel = self._compute_features(pcm)
        self.saver = tf.train.import_meta_graph(self.checkpoint,input_map={'input_tensor:0':log_mel})
        return self.saver

        
    def get_logits(self):
        return tf.get_default_graph().get_tensor_by_name('dense_1/BiasAdd:0')

    def get_probs(self):                
        return tf.get_default_graph().get_tensor_by_name('dense_1/Softmax:0')

class CochlearModel():

    def __init__(self,checkpoint):
        self.checkpoint = checkpoint

    def build_graph(self,pcm):
        self.saver = tf.train.import_meta_graph(self.checkpoint,input_map={'input_1:0':pcm})
        return self.saver

        
    def get_logits(self):
        return tf.get_default_graph().get_tensor_by_name('average_1/truediv:0')
    def get_probs(self):                
        return tf.get_default_graph().get_tensor_by_name('average_1/truediv:0')

