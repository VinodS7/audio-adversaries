from __future__ import absolute_import,print_function,division

import tensorflow as tf
import numpy as np
import time

import deepfool_main
import carlini_wagner_main

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('function','deepfool','function to be called')
flags.DEFINE_string('model_path','task2/save_tensorflow/temp','Tensorflow model path')
flags.DEFINE_string('audio_path','task2/split/training','directory for audio to perform adversarial attacks on')
flags.DEFINE_string('metadata_path','metadata/training.csv','Path to list of file names and labels')
flags.DEFINE_string('adv_audio_path','adv-audio/gcnn','path to save adversarial audio data')
flags.DEFINE_string('exp_data_path','metadata/attack_training.csv','Save data from the experiment')
flags.DEFINE_string('model_type','cochlear','pick from repository the model comes from')
flags.DEFINE_integer('save_data',0,'Whether save data or not')
flags.DEFINE_integer('is_targeted',0,'Is the attack targeted or not')

start = time.process_time()
print('Started timer')
if(FLAGS.function == 'deepfool'):
    if(not(FLAGS.is_targeted)):
        if(FLAGS.model_type  == 'iqbal'):
            deepfool_main.deepfoolattack(FLAGS.audio_path,FLAGS.metadata_path,FLAGS.model_path,FLAGS.exp_data_path,FLAGS.adv_audio_path,FLAGS.save_data)
        elif(FLAGS.model_type == 'cochlear'):
            deepfool_main.deepfoolcochlear(FLAGS.audio_path,FLAGS.metadata_path,FLAGS.model_path,FLAGS.exp_data_path,FLAGS.adv_audio_path,FLAGS.save_data)
 
    else:
        print('Can\'t do targeted with deepfool')

elif(FLAGS.function == 'carlini'):
    if(FLAGS.is_targeted):
        print('Performing targeted attack')
        carlini_wagner_main.carliniwagnertargeted(FLAGS.audio_path,FLAGS.metadata_path,FLAGS.model_path,FLAGS.exp_data_path,FLAGS.adv_audio_path,FLAGS.save_data)
    elif(not(FLAGS.is_targeted)):
        print('Untargeted attack')
        if(FLAGS.model_type == 'iqbal'):
            carlini_wagner_main.carliniwagneruntargeted(FLAGS.audio_path,FLAGS.metadata_path,FLAGS.model_path,FLAGS.exp_data_path,FLAGS.adv_audio_path,FLAGS.save_data)
        elif(FLAGS.model_type == 'cochlear'):
            carlini_wagner_main.carliniwagneruntargetedcochlear(FLAGS.audio_path,FLAGS.metadata_path,FLAGS.model_path,FLAGS.exp_data_path,FLAGS.adv_audio_path,FLAGS.save_data)
 
end = time.process_time()
print(end-start)
