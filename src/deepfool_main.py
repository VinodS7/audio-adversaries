from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import librosa
from tqdm import tqdm
import time

import config as cfg
import file_io as io
import utils_tf
import utils

from attacks import deep_fool_method as DFM
from cleverhans_wrapper import CleverHansModel


def deepfoolattack(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    print(save_data)
    x,_ = utils_tf._load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    df = pd.read_csv(metadata_path)
    label_names= df.iloc[:,2].values
    file_names = df.iloc[:,1].values
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
    sample_rate = 32000
    
    audio_name = []
    audio_length = []
    original_label = []
    original_confidence = []
    new_label = []
    new_confidence = []
    new_o_label_conf = []
    snr = []
    with tf.Graph().as_default() as graph:
        mel_filt = tf.convert_to_tensor(mel_fb,dtype=tf.float32)
        model = CleverHansModel(model_path +'.meta',sample_rate,generator,mel_filt)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        saver= model.build_graph(pcm)
        deepfool = DFM.DeepFool(model)
        deepfool.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error")
 
            labels= utils_tf._convert_label_name_to_label(label_names[i])
            s = sess.run([model.get_probs()],feed_dict={'input_audio:0':data})
            
            s = np.squeeze(s)
            if (s.ndim != 1):
                s = np.mean(s,axis=0)
                      
                
            print('Original label number:',np.argmax(s))
            print('Original label confidence:',np.max(s))
                
            tic = time.process_time()
            adv = deepfool.attack(sess,data,int(q))
            toc = time.process_time()

            print('Time for processing sample:',toc-tic,'for iteration:',i)
            preds = sess.run([model.get_probs()],feed_dict={pcm:adv})
            preds = np.squeeze(preds)

            if(preds.ndim !=1):
                preds = np.mean(preds,axis=0)
            print('New label number:',np.argmax(preds))
            print('New label confidence:',np.max(preds))
                
            if(save_data):
                librosa.output.write_wav(adv_audio_path + 'adv-' + audio_file_name,adv,sample_rate)
                
            audio_name.append(audio_file_name)
            audio_length.append(int(q))
            original_label.append(np.argmax(s))
            original_confidence.append(np.max(s))
            new_label.append(np.argmax(preds))
            new_confidence.append(np.max(preds))
            new_o_label_conf.append(preds[np.argmax(s)])
            snr.append(20*np.log10(np.mean(data**2)/(np.mean((adv-data)**2))))
        if(save_data):
            df_deepfool = pd.DataFrame({'audio_name':audio_name,'audio_length':audio_length,'original_label':original_label,'original_confidence':original_confidence,'new_label':new_label,'new_confidence':new_confidence,'new_orig_conf':new_o_label_conf,'SNR':snr})
        
            with open(exp_data_path,'a') as f:
                df_deepfool.to_csv(f,header=False)


