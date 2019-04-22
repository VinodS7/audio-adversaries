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

from attacks import carlini_wagner_method as CW
from cleverhans_wrapper import CleverHansModel,CochlearModel

def carliniwagnertargeted(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    x,_ = utils_tf._load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    df = pd.read_csv(metadata_path)
    gt_labels= df.iloc[:,2].values
    file_names = df.iloc[:,1].values
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
    sample_rate = 32000
    label_list = ["Bass_drum", "Cello", "Clarinet", "Oboe", "Snare_drum", "Violin_or_fiddle"]
 

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
        carliniwagner = CW.CarliniWagnerAttack(model,learning_rate = 1e-5,confidence=500,targeted=True,max_iterations=1000,binary_search_steps = 2)
        saver = carliniwagner.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error")
 
            gt_label = gt_labels[i]
            
            print('Ground truth label:',gt_label,'Audio_file:',file_names[i]) 
            for l in range(len(label_list)): 
                label = utils_tf._convert_label_name_to_label(label_list[l])
                if(label == gt_label):
                    continue
                
                adv,o_label,o_conf,n_label,n_conf,n_gt_conf = carliniwagner.attack(sess,data,label,np.repeat(label,int(q)),int(q),prob_thresh=0.975)
                
                if(save_data):
                    librosa.output.write_wav(adv_audio_path + 'adv-' +label_list[l] + '-'+ audio_file_name,adv,sample_rate)
            
                audio_name.append(audio_file_name)
                audio_length.append(int(q))
                original_label.append(o_label)
                original_confidence.append(o_conf)
                new_label.append(n_label)
                new_confidence.append(n_conf)
                new_o_label_conf.append(n_gt_conf)
    
                snr.append(10*np.log10(np.mean(data**2)/(np.mean(adv-data)**2)))
        if(save_data):
            df_cw = pd.DataFrame({'audio_name':audio_name,'audio_length':audio_length,'original_label':original_label,'original_confidence':original_confidence,'new_label':new_label,'new_confidence':new_confidence,'SNR':snr},columns=['audio_name','audio_length','original_label','original_confidence','new_label','new_confidence','SNR'])
        
            with open(exp_data_path,'a') as f:
                df_cw.to_csv(f)

def carliniwagnertargetedcochlear(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    df = pd.read_csv(metadata_path)
    gt_labels= df.iloc[:,2].values
    file_names = df.iloc[:,1].values
    sample_rate = 32000
    label_list = ["Bass_drum", "Cello", "Clarinet", "Oboe", "Snare_drum", "Violin_or_fiddle"]
 

    audio_name = []
    audio_length = []
    original_label = []
    original_confidence = []
    new_label = []
    new_confidence = []
    new_o_label_conf = []
    snr = []
    with tf.Graph().as_default() as graph:
        model = CochlearModel(model_path +'.meta')
        pcm = tf.placeholder(tf.float32,shape=[None,None],name='input_audio')
        carliniwagner = CW.CarliniWagnerAttack(model,learning_rate = 1e-5,confidence=500,targeted=True,max_iterations=2000,binary_search_steps = 1,abort_early=False)
        saver = carliniwagner.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
                data = np.expand_dims(data,axis=0)
            except EOFError:
                print("EOF Error")
 
            gt_label= gt_labels[i]
            
            print('Ground truth label:',gt_label,'File_name:',file_names[i]) 
            for l in range(len(label_list)): 
                label = utils_tf._convert_label_name_to_label(label_list[l])
                if(label == gt_label):
                    continue
                
                adv,o_label,o_conf,n_label,n_conf,n_gt_conf = carliniwagner.attack(sess,data,label,np.repeat(label,int(q)),int(q),prob_thresh=0.975)
                adv = np.squeeze(adv)
                if(save_data):
                    librosa.output.write_wav(adv_audio_path + 'adv-' + label_list[l] + '-'+audio_file_name,adv,sample_rate)
            
                audio_name.append(audio_file_name)
                audio_length.append(int(q))
                original_label.append(o_label)
                original_confidence.append(o_conf)
                new_label.append(n_label)
                new_confidence.append(n_conf)
                new_o_label_conf.append(n_gt_conf)
    
                snr.append(10*np.log10(np.mean(data**2)/(np.mean(adv-data)**2)))
        if(save_data):
            df_cw = pd.DataFrame({'audio_name':audio_name,'audio_length':audio_length,'original_label':original_label,'original_confidence':original_confidence,'new_label':new_label,'new_confidence':new_confidence,'SNR':snr},columns=['audio_name','audio_length','original_label','original_confidence','new_label','new_confidence','SNR'])
        
        
            with open(exp_data_path,'a') as f:
                df_cw.to_csv(f)

def carliniwagneruntargeted(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
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
        carliniwagner = CW.CarliniWagnerAttack(model,learning_rate = 1e-5,initial_const = 1e-2,max_iterations=1000,confidence = 500,binary_search_steps = 2)
        saver = carliniwagner.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error")
 
            label= utils_tf._convert_label_name_to_label(label_names[i])
             
            print('Ground truth label:',label_names[i])
                
                 
         
            labels_batchwise = np.repeat(label,int(q))
            
            tic = time.process_time()
            adv,o_label,o_conf,n_label,n_conf,n_conf_gt = carliniwagner.attack(sess,data,label,labels_batchwise,int(q),prob_thresh = 0.0244)
            toc = time.process_time()

            print('Time for iteration:',i,'is',toc-tic)
            if(save_data):
                librosa.output.write_wav(adv_audio_path + 'adv-' + audio_file_name,adv,sample_rate)
            
            audio_name.append(audio_file_name)
            audio_length.append(int(q))
            original_label.append(o_label)
            original_confidence.append(o_conf)
            new_label.append(n_label)
            new_confidence.append(n_conf)
            new_o_label_conf.append(n_conf_gt)
            snr.append(10*np.log10(np.mean(data**2)/(np.mean((adv-data)**2))))
        if(save_data):
            df_cw = pd.DataFrame({'audio_name':audio_name,'audio_length':audio_length,'original_label':original_label,'original_confidence':original_confidence,'new_label':new_label,'new_confidence':new_confidence,'new_orig_conf':new_o_label_conf,'SNR':snr})
        
            with open(exp_data_path,'w') as f:
                df_cw.to_csv(f,header=False)

def carliniwagneruntargetedcochlear(audio_path,metadata_path,model_path,exp_data_path,adv_audio_path,save_data=False):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    df = pd.read_csv(metadata_path)
    label_names= df.iloc[:,2].values
    file_names = df.iloc[:,1].values
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
        model = CochlearModel(model_path +'.meta')
        pcm = tf.placeholder(tf.float32,shape=[None,None],name='input_audio')
        carliniwagner = CW.CarliniWagnerAttack(model,learning_rate = 1e-5,initial_const = 1e-2,max_iterations=1000,confidence = 500,binary_search_steps = 2)
        saver = carliniwagner.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,model_path)
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = utils_tf._preprocess_data(audio_path,audio_file_name)
                data = np.expand_dims(data,axis=0)
            except EOFError:
                print("EOF Error")
 
            label= utils_tf._convert_label_name_to_label(label_names[i])
             
            print('Ground truth label:',label_names[i],label)
                
                 
         
            
            tic = time.process_time()
            adv,o_label,o_conf,n_label,n_conf,n_conf_gt = carliniwagner.attack(sess,data,label,label,1,prob_thresh = 0.0244)
            toc = time.process_time()

            print('Time for iteration:',i,'is',toc-tic)
            adv = np.squeeze(adv)
            if(save_data):
                librosa.output.write_wav(adv_audio_path + 'adv-' + audio_file_name,adv,sample_rate)
            
            audio_name.append(audio_file_name)
            audio_length.append(int(q))
            original_label.append(o_label)
            original_confidence.append(o_conf)
            new_label.append(n_label)
            new_confidence.append(n_conf)
            new_o_label_conf.append(n_conf_gt)
            snr.append(10*np.log10(np.mean(data**2)/(np.mean((adv-data)**2))))
        if(save_data):
            df_cw = pd.DataFrame({'audio_name':audio_name,'audio_length':audio_length,'original_label':original_label,'original_confidence':original_confidence,'new_label':new_label,'new_confidence':new_confidence,'new_orig_conf':new_o_label_conf,'SNR':snr})
        
            with open(exp_data_path,'w') as f:
                df_cw.to_csv(f,header=False)


