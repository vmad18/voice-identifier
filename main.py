'''
@author v18
'''

from model.training import f_train
from utils.PostProcess import PostProcess
from utils.PreProcess import PreProcess

voice_identifier = f_train()


postp = PostProcess(voice_identifier.generatevf())

'''
Record Voice Samples
'''

name = input("Enter the name of the first voice: ").lower()
while not (name is "skip!"):
    postp.addNewVoice(name)
    name = input("Enter the name of the next voice: ").lower()


'''
OR Add Voice from File Path (.wav, .mp3, .flac)
'''
path = ""
name = ""
postp.addVoicesFromPath(path, name, add=True) 


'''
Generate Clusters & Predicting
'''
postp.generateCluster()
postp.predict(postp.record(rc=2)) #Predict Live Audio

path = ""
postp.predictFromPath(path, thresh=-15) #Predict from Audio File

