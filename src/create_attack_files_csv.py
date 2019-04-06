import pandas as pd
import numpy as np

def _convert_label_name_to_label(label_name):
    label=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
                
    return label.index(label_name)


df_new = pd.DataFrame(columns=['fname','label'])

metadata_file = 'metadata/training.csv'
save_file = 'metadata/attack_training.csv'
df = pd.read_csv(save_file)

count = np.zeros(41)
#for iter in range(df.shape[0]):
#    if(df.loc[iter]['manually_verified']):
#        label = _convert_label_name_to_label(df.loc[iter]['label'])
#        count[label]+=1
#        if(count[label]<11):
#            df_new = df_new.append({'fname':df.loc[iter]['fname'],'label':df.loc[iter]['label']},ignore_index=True)

#print(df_new)

#with open(save_file,'w') as f:
#    df_new.to_csv(f,header=False)

for iter in range(df.shape[0]):
    label = _convert_label_name_to_label(df.iloc[iter][2])
    count[label]+=1

print(count,np.sum(count))

