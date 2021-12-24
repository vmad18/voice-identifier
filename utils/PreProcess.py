import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import librosa

class PreProcess:

  def __init__(self):
    self.feats = []
    self.y = []
    self.labels = []
    self.paths = []
    self.people = []

  def getPath(self, path):
    for c in os.listdir(path):
      self.paths.append(path+c)

  def clearPaths(self):
    self.paths = []
  
  def process1(self):
    for p in self.paths: 
      for f in os.listdir(p):
        self.labels.append(f)
        self.people.append(f)
        for noob in os.listdir(p+"/"+f):
          if(noob.find(".flac") == -1): continue
          fpath = p+'/'+f+'/'+noob 
          self.feats.append(PreProcess.features_extractor(fpath))
          self.y.append(f)
  
  def createDataSet(self):
    if len(self.feats) == 0 or len(self.y) == 0: return None, None, None, None
    n_feats = np.asarray(self.feats)
    n_person = np.asarray(self.y)
    
    le = LabelEncoder()
    y_person = to_categorical(le.fit_transform(n_person))
    
    return train_test_split(n_feats, 
                            y_person, 
                            test_size=0.2, 
                            random_state=0)

  @staticmethod
  def features_extractor(file):
      audio, sample_rate = librosa.load(
                                        file, 
                                        res_type="kaiser_fast") 
      
      audiot, index = librosa.effects.trim(audio, top_db=20, frame_length = 1, hop_length = 1)

      mfccs_features = librosa.feature.mfcc(y=audiot, sr=sample_rate, n_mfcc=40)
      mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
      return mfccs_scaled_features

    