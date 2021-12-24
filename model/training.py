import sys 

sys.path.append("/VoiceRecognition/model")
sys.path.append("/VoiceRecognition/utils")

import os
from utils.PreProcess import PreProcess
from model.Model import VoiceRecognition
from tensorflow.keras.models import load_model 


train_model = input("Use Pretrained Model? ").lower()
while(train_model != "yes" and train_model != "no"):
    train_model = input("Type 'yes' or 'no' ").lower()

'''
Select path for training
'''

voices1, not_voices = "VoiceRecognition/Datasets", "VoiceRecognition/Datasets"
epochs, bs = 200, 32
sp = "VoiceRecognition/model/voice_identifier.hdf5"
mod = None

pp = PreProcess()


if train_model == "no":
    pp.getPath(voices1)
    pp.process1()

    pp.labels.append("notvoice")
    for p in os.lisdir(not_voices):
        pp.feats.append(PreProcess.features_extractor(not_voices + p))
        pp.y.append("notvoice")
else:
    mod = load_model("model/voice_identifier.hdf5")

x_train, y_train, x_val, y_val = pp.createDataSet()

def f_train() -> VoiceRecognition:
    vrec = VoiceRecognition(
                            epochs, 
                            bs, 
                            pp.feats, 
                            pp.labels, 
                            x_train, 
                            y_train, 
                            x_val, 
                            y_val, 
                            sp)
    if mod is None:
        vrec.generate()
        vrec.train()
        vrec.eval()
        return vrec 

    vrec.load_model(mod)
    return vrec
