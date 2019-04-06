from __future__ import absolute_import,print_function,division


import os

import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import keras.models
from keras import backend as K
from tqdm import tqdm

import config as cfg
import utils
import file_io as io
from attacks import fast_gradient_method as FGM
from attacks import projected_gradient_method as PGM
from attacks import baseline_white_noise as BWN
from attacks import lbfgs as LB
from attacks import deep_fool_method as DFM
from attacks import saliency_map_method as SM
from attacks import carlini_wagner_method as CW


from cleverhans_wrapper import CleverHansModel
from gated_conv import GatedConv

def save_keras_model_as_tensorflow_meta_graph(keras_model_path,save_model_path):
    custom_objects = {'GatedConv':GatedConv}
    m = keras.models.load_model(keras_model_path,custom_objects)
    graph = K.get_session().graph
    sess = K.get_session() 
    saver = tf.train.Saver()
    saver.save(sess,save_model_path)

    return

def compare_features(audio_path,audio_number,metadata_path,save_path,keras_model_path):
    
    m = keras.models.load_model(keras_model_path)
    #Load dataset
    x,df = _load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    file_names = df.index.to_list()
    audio_file_name = file_names[audio_number]
    print(audio_file_name)
    path = os.path.join(audio_path,audio_file_name)
    x,fs = librosa.load(path,sr=None)
    print('Sample rate:',fs)
    print('length in sec:',float(x.shape[0]/fs))
    #Librosa features
    
    x = librosa.resample(x,fs,32000)
    x_tf = x
    D = librosa.stft(x,n_fft=1024,hop_length=512)
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64)
    S = np.dot(mel_fb,np.abs(D)**2).T
    feat = librosa.power_to_db(S,ref=np.max,top_db=None)
    spec_test = generator.standardize(_reshape_spec(feat))
    graph = tf.Graph()
    with graph.as_default():
        pcm = tf.placeholder(tf.float32,shape=[None])
        filter_banks = tf.placeholder(tf.float32,shape=[None,None])
        stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=512,
                                           fft_length=1024,pad_end=True)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0, 16000, 64
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
              num_mel_bins, num_spectrogram_bins, 32000, lower_edge_hertz,
                upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
              tf.pow(spectrograms,2), filter_banks, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
         linear_to_mel_weight_matrix.shape[-1:]))
        
        max_val = tf.reduce_max(mel_spectrograms,axis=None)
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = 10*((tf.log(mel_spectrograms + 1e-6)-tf.log(max_val+1e-6))/tf.log(tf.constant(10,dtype=tf.float32)))

    with tf.Session(graph=graph) as sess:
        lms = sess.run([log_mel_spectrograms],feed_dict={pcm:x_tf,filter_banks:mel_fb.T})
        lms = np.squeeze(lms)
        spec_tf = generator.standardize(_reshape_spec(lms,32))


    p = m.predict(spec_tf[:,:,:,np.newaxis])
    print(np.argmax(p,axis=1),np.max(p,axis=1))
       
def experiment1(audio_path,audio_number,metadata_path,save_path,save_data_path):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    
    x,df = _load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    file_names = df.index.to_list()
    audio_file_name = file_names[audio_number]
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
    sample_rate = 32000
    
    #df_fgsm = pd.DataFrame(columns=['audio_name','audio_length','original_label','original_confidence','new_label','new_confidence','mean_absolute_error'])
    #df_fgsm.to_csv('adv-audio/vgg13/fgsm.csv')
    #df_baseline = pd.DataFrame(columns=['audio_name','audio_length','original_label','original_confidence','new_label','new_confidence','mean_absolute_error'])
    #df_baseline.to_csv = ('adv-audio/vgg13/baseline.csv')
    with tf.Graph().as_default() as graph:
        mel_filt = tf.convert_to_tensor(mel_fb,dtype=tf.float32)
        model = CleverHansModel(save_path +'.meta',sample_rate,generator,mel_filt)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        saver = model.build_graph(pcm)
        pgd = PGM.ProjectedGradientDescent(model,rms_ratio = 6)
        pgd.build_attack(pcm)
        bwn = BWN.BaselineWhiteNoise(model,rms_ratio = 6)
        bwn.build_attack(pcm)

    with tf.Session(graph=graph) as sess:
        saver.restore(sess,save_path)
        for i in range(len(file_names)):
            audio_file_name = file_names[i]
            try:
                data,q = _preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error dammit")
            label_name = _get_label_from_audio(audio_path,audio_file_name,metadata_path)
            labels= _convert_label_name_to_label(label_name)
            s = sess.run([model.get_probs()],feed_dict={'input_audio:0':data})
            
            s = np.squeeze(s)
            if (s.ndim != 1):
                s = np.max(s,axis=0)
                      
            if(np.argmax(s) == labels):
                
                print('Iteration number:',i)
                print('Original label number:',np.argmax(s))
                print('Original label confidence:',np.max(s))
                labels = np.repeat(labels,int(q))
                adv,mae,label,confidence = pgd.attack(data,labels,1,sess)
                librosa.output.write_wav(save_data_path +'fgsm/'+audio_file_name[:-4]+'-adv.wav',adv,sample_rate)
                df_fgsm = pd.DataFrame(columns=['audio_name','audio_length','original_label','original_confidence','new_label','new_confidence','mean_absolute_error'])

                df_fgsm = df_fgsm.append({'audio_name':audio_file_name,'audio_length':data.shape[0],'original_label':np.argmax(s),'original_confidence':np.max(s),'new_label':label,'new_confidence':confidence,'mean_absolute_error':mae},ignore_index=True) 
                with open(save_data_path +'fgsm.csv','a') as f:
                    df_fgsm.to_csv(f,header=False)
                
                
                adv,mae,label,confidence = bwn.attack(data,sess)
                librosa.output.write_wav(save_data_path +'baseline/'+audio_file_name[:-4]+'-adv.wav',adv,sample_rate)
                df_baseline = pd.DataFrame(columns=['audio_name','audio_length','original_label','original_confidence','new_label','new_confidence','mean_absolute_error'])
                df_baseline = df_baseline.append({'audio_name':audio_file_name,'audio_length':data.shape[0],'original_label':np.argmax(s),'original_confidence':np.max(s),'new_label':label,'new_confidence':confidence,'mean_absolute_error':mae},ignore_index=True)     
                with open(save_data_path +'baseline.csv','a') as f:
                    df_baseline.to_csv(f,header=False)


def experiment2(audio_path,audio_number,metadata_path,save_path):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    
    x,df = _load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    file_names = df.index.to_list()
    audio_file_name = file_names[audio_number]
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64).T
    sample_rate = 32000

 
    with tf.Graph().as_default() as graph:
        mel_filt = tf.convert_to_tensor(mel_fb,dtype=tf.float32)
        model = CleverHansModel(save_path +'.meta',sample_rate,generator,mel_filt)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        saver= model.build_graph(pcm)
        saliencymap = SM.SaliencyMapMethod(model,41)
        saliencymap.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,save_path)
        for i in range(111,112):
            audio_file_name = file_names[i]
            try:
                data,q = _preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error dammit")
            label_name = _get_label_from_audio(audio_path,audio_file_name,metadata_path)
            labels= _convert_label_name_to_label(label_name)
            s = sess.run([model.get_probs()],feed_dict={'input_audio:0':data})
            
            s = np.squeeze(s)
            if (s.ndim != 1):
                s = np.mean(s,axis=0)
                      
            if(np.argmax(s) == labels):
                
                print('Iteration number:',i)
                print('Original label number:',np.argmax(s))
                print('Original label confidence:',np.max(s))
                labels = np.repeat(20,int(q))
                adv = saliencymap.attack(data,labels,sess)
                
                preds = sess.run([model.get_probs()],feed_dict={pcm:adv})
                preds = np.squeeze(preds)
                if(preds.ndim==1):
                    print(np.max(preds),np.argmax(preds))
                else:
                    print(np.max(preds,axis=1),np.argmax(preds,axis=1))

                if(preds.ndim !=1):
                    preds = np.mean(preds,axis=0)

                print(np.argmax(preds),np.max(preds))

                librosa.output.write_wav('adv-cw.wav',adv,sample_rate)
                librosa.output.write_wav('original-cw.wav',data,sample_rate)

def deepfoolattack(audio_path,adversarial_data_path,metadata_path,save_path,save_metadata,save_audio_path):
    #Run the attacks to generate adversarial attacks on manually verified examples on the training and test data
    #Load dataset to normalize new data
    
    x,_ = _load_dataset(cfg.to_dataset('training'))
    generator = utils.fit_scaler(x)
    df = pd.read_csv(adversarial_data_path)
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
    mean_sq_error = []
    std_dev = []
    snr = []
    with tf.Graph().as_default() as graph:
        mel_filt = tf.convert_to_tensor(mel_fb,dtype=tf.float32)
        model = CleverHansModel(save_path +'.meta',sample_rate,generator,mel_filt)
        pcm = tf.placeholder(tf.float32,shape=[None],name='input_audio')
        saver= model.build_graph(pcm)
        deepfool = DFM.DeepFool(model)
        deepfool.build_attack(pcm)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess,save_path)
        for i in range(df.shape[0]):
            audio_file_name = file_names[i]
            try:
                data,q = _preprocess_data(audio_path,audio_file_name)
            except EOFError:
                print("EOF Error dammit")
 
            labels= _convert_label_name_to_label(label_names[i])
            s = sess.run([model.get_probs()],feed_dict={'input_audio:0':data})
            
            s = np.squeeze(s)
            if (s.ndim != 1):
                s = np.mean(s,axis=0)
                      
            if(np.argmax(s) == labels):
                
                print('Iteration number:',i)
                print('Original label number:',np.argmax(s))
                print('Original label confidence:',np.max(s))
                

                adv = deepfool.attack(sess,data,int(q))
                
                preds = sess.run([model.get_probs()],feed_dict={pcm:adv})
                preds = np.squeeze(preds)

                if(preds.ndim !=1):
                    preds = np.mean(preds,axis=0)

                print('New label number:',np.argmax(preds))
                print('New label confidence:',np.max(preds))
                
                librosa.output.write_wav(save_audio_path + 'adv-' + audio_file_name,adv,sample_rate)
                audio_name.append(audio_file_name)
                audio_length.append(int(q))
                original_label.append(np.argmax(s))
                original_confidence.append(np.max(s))
                new_label.append(np.argmax(preds))
                new_confidence.append(np.max(preds))
                mean_sq_error.append(np.mean((adv-data)**2))
                std_dev.append(np.std((adv-data)**2))
                snr.append(20*np.log10(np.mean(data**2)/(np.mean(adv-data)**2)))
        
        df_deepfool = pd.DataFrame({'audio_name':audio_name,'audio_length':audio_length,'original_label':original_label,'original_confidence':original_confidence,'new_label':new_label,'new_confidence':new_confidence,'mean_square_error':mean_sq_error,'standard_deviation':std_dev,'SNR':snr})
        with open(save_metadata +'deepfool-gcnn.csv','a') as f:
            df_deepfool.to_csv(f,header=False)



def _load_dataset(dataset):
    """Load input data and the associated metadata for a dataset.

    Args:
        dataset: Structure encapsulating dataset information.

    Returns:
        tuple: Tuple containing:

            x (np.ndarray): The input data of the dataset.
            df (pd.DataFrame): The metadata of the dataset.
    """
    import features

    # Load feature vectors and reshape to 4D tensor
    features_path = os.path.join(cfg.extraction_path, dataset.name + '.h5')
    x, n_chunks = utils.timeit(lambda: features.load_features(features_path),
                               'Loaded features of %s dataset' % dataset.name)
    x = np.expand_dims(x, -1)
    assert x.ndim == 4

    # Load metadata and duplicate entries based on number of chunks
    df = io.read_metadata(dataset.metadata_path)

    return x, df



def _preprocess_data(audio_path,audio_file_name):
    filelist = os.listdir(audio_path)
    print('Audio file name:', audio_file_name)
    

    audio_file_path = os.path.join(audio_path,audio_file_name)
    data,q = _load_audio(audio_file_path)
    return data,q

def _get_label_from_audio(audio_path,audio_file_name,metadata_path):
    filelist = os.listdir(audio_path)
    df = pd.read_csv(metadata_path)
    for iter in range(df.shape[0]):
        temp = df.loc[iter]['fname']
        temp = temp[:-4]

        if(temp in audio_file_name):
            return df.loc[iter]['label']

def _convert_label_name_to_label(label_name):
    label=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
                
    return label.index(label_name)
def _convert_label_to_label_name(iter):
    label=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
    label_name = label[iter]
    return label_name

def  _load_audio(audio_file_path):
    data,fs = librosa.load(audio_file_path)
    data = librosa.resample(data,fs,32000)
    l = 128*512
    r = data.shape[0]%l
    q = np.ceil(data.shape[0]/l)
    if (r==0):
        return data,q
    else:
        return np.pad(data,(0,l-r),'constant',constant_values=(0)),q


def _compute_features(pcm,sample_rate):
    
    stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=512,
                                           fft_length=1024,pad_end=True)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0, 16000, 64
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
              num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
              tf.pow(spectrograms,2), linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
         linear_to_mel_weight_matrix.shape[-1:]))

#    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = 10*tf.log(mel_spectrograms + 1e-6)
    log_mel_spectrograms = tf.reshape(log_mel_spectrograms,shape=[-1,128,64,1])
 


    return log_mel_spectrograms

def _compute_librosa_features(pcm,sample_rate):
    pcm = librosa.resample(pcm,sample_rate,32000)
    D = librosa.stft(pcm,n_fft=1024,hop_length=512)
    mel_fb = librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64)
    S = np.dot(mel_fb,np.abs(D)**2).T
    return librosa.power_to_db(S,ref=np.max,top_db=None)

def _reshape_spec(feat,r_threshold=32):
    q = feat.shape[0] // 128
    r = feat.shape[0] % 128
    r_threshold = 32
    print(q,r)

    if not q:
        split = [utils.pad_truncate(feat, 128, pad_value=np.min(feat))]
    else:
        off = r // 2 if r < r_threshold else 0
        split = np.split(feat[off:q * 128 + off], q)
        if r >= r_threshold:
            split.append(feat[-128:])
        return np.array(split)


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('function','save','function to be called')
flags.DEFINE_string('keras_path','task2/models/jul31_vgg13/all/model.40-0.8708.h5','Keras model full path')
flags.DEFINE_string('save_path','task2/save_tensorflow/temp','Tensorflow save path')
flags.DEFINE_string('audio_dir','task2/split/training','directory for audio')
flags.DEFINE_string('metadata_dir','metadata/training.csv','Path to training or testing file')
flags.DEFINE_string('save_data_path','adv-audio/gcnn','path to save data')
flags.DEFINE_integer('audio_number',0,'Audio number from audio directory')
flags.DEFINE_string('adversary_data','metadata/attack_training.csv','blah')
flags.DEFINE_string('save_metadata','experiment_data/','Path to save experiment metadata')
if(FLAGS.function == 'save'):
    save_keras_model_as_tensorflow_meta_graph(FLAGS.keras_path,FLAGS.save_path)
elif(FLAGS.function == 'predict'):
    predict_tf(FLAGS.audio_dir,FLAGS.audio_number,FLAGS.metadata_dir,FLAGS.save_path,FLAGS.keras_path)
elif(FLAGS.function == 'adversary'):
    generate_adversary(FLAGS.audio_dir,FLAGS.audio_number,FLAGS.metadata_dir,FLAGS.save_path)
elif(FLAGS.function == 'experiment1'):
    experiment1(FLAGS.audio_dir,FLAGS.audio_number,FLAGS.metadata_dir,FLAGS.save_path,FLAGS.save_data_path)
elif(FLAGS.function == 'compare'):
    compare_features(FLAGS.audio_dir,FLAGS.audio_number,FLAGS.metadata_dir,FLAGS.save_path,FLAGS.keras_path)
elif(FLAGS.function == 'experiment2'):
    experiment2(FLAGS.audio_dir,FLAGS.audio_number,FLAGS.metadata_dir,FLAGS.save_path)
elif(FLAGS.function == 'deepfool'):
    deepfoolattack(FLAGS.audio_dir,FLAGS.adversary_data,FLAGS.metadata_dir,FLAGS.save_path,FLAGS.save_metadata,FLAGS.save_data_path)
