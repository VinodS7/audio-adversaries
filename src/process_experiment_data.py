
from __future__ import absolute_import, print_function,division

import tensorflow as tf
import numpy as np
import pandas as pd


df = pd.read_csv('experiment_data/deepfool.csv')

original_labels = df.iloc[:,3].to_list()
new_labels = df.iloc[:,5].to_list()
snr = df.iloc[:,-1].to_list()
print(np.mean(snr),np.min(snr),np.max(snr))
count = 0
for i in range(len(new_labels)):
    if(original_labels[i] == new_labels[i]):
        count+=1


print (count)
