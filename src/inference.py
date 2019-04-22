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

from cleverhans_wrapper import CleverHansModel,CochlearModel


def inferenceiqbal(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    x,_ = utils_tf._load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    df = pd.read_csv(metadata_path)
    label_names= df.iloc[:,1].values
    file_names = df.iloc[:,0].values
    
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
    sample_rate = 32000
   
    audio_name = []
    ground_truth = []
    inferred_label = []
    inferred_confidence = []
    with tf.Graph().as_default() as graph:
        mel_filt = tf.convert_to_tensor(mel_fb,dtype=tf.float32)
        model = CleverHansModel(model_path +'.meta',sample_rate,generator,mel_filt)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        saver= model.build_graph(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        count = 0
        count_tot = 0
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error")
 
            gt_label= utils_tf._convert_label_name_to_label(label_names[i])
            s = sess.run([model.get_probs()],feed_dict={'input_audio:0':data})
            
            s = np.squeeze(s)
            if (s.ndim != 1):
                s = np.mean(s,axis=0)
            label = np.argmax(s)
            count_tot +=1
            if(label == gt_label):
                count +=1
            
            if(i%1000 == 0):
                print('Iteration number:',i)
                print('Current accuracy:',float(count/count_tot))
            audio_name.append(audio_file_name)
            ground_truth.append(gt_label)
            inferred_label.append(label)
            inferred_confidence.append(np.max(s))
        if(save_data):
            df_deepfool = pd.DataFrame({'audio_name':audio_name,'ground_truth':ground_truth,'inferred_label':inferred_label,'inferred_confidence':inferred_confidence})
        
            with open(exp_data_path,'w') as f:
                df_deepfool.to_csv(f,header=False)


def inferencecochlear(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    df = pd.read_csv(metadata_path)
    label_names= df.iloc[:,1].values
    file_names = df.iloc[:,0].values
    sample_rate = 32000
    
    audio_name = []
    inferred_label = []
    inferred_confidence = []
    ground_truth = []
    with tf.Graph().as_default() as graph:
        model = CochlearModel(model_path +'.meta')
        pcm = tf.placeholder(tf.float32,shape=[None,None],name='input_audio')
        saver= model.build_graph(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        count = 0
        count_tot = 0
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
                data = np.expand_dims(data,axis=0)
            except EOFError:
                print("EOF Error")
 
            gt_label= utils_tf._convert_label_name_to_label(label_names[i])
            s = sess.run([model.get_probs()],feed_dict={'input_audio:0':data})
            
            s = np.squeeze(s)
            if (s.ndim != 1):
                s = np.mean(s,axis=0)
            
            count_tot+=1  

            if(gt_label == np.argmax(s)):
                count +=1
            
            if(i%1000 == 0):
                print('Iteration number:',i)
                print('Current accuracy:',float(count/count_tot))
            
            audio_name.append(audio_file_name)
            inferred_label.append(np.argmax(s))
            inferred_confidence.append(np.max(s))
            ground_truth.append(gt_label)
        if(save_data):
            df_infer = pd.DataFrame({'audio_name':audio_name,'ground_truth':ground_truth,'inferred_label':inferred_label,'inferred_confidence':inferred_confidence})
        
            with open(exp_data_path,'w') as f:
                df_infer.to_csv(f,header=False)


def evaluate(metadata_path,inference_path):
    df_meta = pd.read_csv(metadata_path)
    df_infer = pd.read_csv(inference_path)
    manually_verified = df_meta.iloc[:,2].to_list()
    ground_truth = df_infer.iloc[:,2].to_list()
    inferred_label = df_infer.iloc[:,3].to_list()
    count_tot = 0
    count_true = 0
    confusion_matrix = np.zeros([41,41])
    for i in range(df_infer.shape[0]):
        if(manually_verified[i]):
            count_tot +=1
            if(ground_truth[i] == inferred_label[i]):
                count_true +=1
            
            confusion_matrix[ground_truth[i],inferred_label[i]] +=1
    print(float(count_true/count_tot))
    print(confusion_matrix)
