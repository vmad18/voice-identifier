from utils.PreProcess import PreProcess

try:
    import pyaudio
except:
    pass 

import wave
import os
import numpy as np
import tensorflow
from sklearn.cluster import KMeans
from time import sleep

class PostProcess:

    def __init__(self, model, g=0):
        self.vf = model
        self.voices = []
        self.groups = g
        self.cluster = None
        self.generated = False
        self.whois = {}
        try:
            self.mic = pyaudio.PyAudio()
        except:
            pass
  
    def remove(self, index = -1):
        self.voices.pop(index)

    def _train(self):
        return np.asarray(self.voices, dtype = "float64")

    def addGroup(self) -> None:
        self.groups+=1

    def remGroup(self) -> None:
        self.groups-=1

    def _wait(self, sec):
        for i in range(sec):
            print("Wait...", (sec - i - 1))
            sleep(1)

    def record(self, rc = 5):
        try:
            CHUNK = 2<<9
            FORMAT = pyaudio.paInt16
            CHANNELS = 2 
            RATE = 44100
            RECORD_SECONDS = rc

    
            stream = self.mic.open(
                                    format = FORMAT, 
                                    rate = RATE, 
                                    channels = CHANNELS, 
                                    input = True, 
                                    frames_per_buffer = CHUNK)

            frames = []
            for i in range(int(RATE/CHUNK * RECORD_SECONDS)):
                frames.append(stream.read(CHUNK))

            stream.stop_stream()
            stream.close()

            output = "utils/audio_out/" + len(os.listdir("utils/audio_out"))-1

            wf = wave.open(output, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.mic.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            features = PreProcess.features_extractor(output)
            os.remove(output)
        except:
            print("PyAudio Not Working")
            return None
        return features

    def addNewVoice(self, name):
        self.addGroup()

        print("1-second sample- Speak...")
        self.voices.append(self.voiceFeatures(self.record(rc = 3)))
        print("...Finished")
        self._wait(2) 

        print("3-second sample- Speak...")
        self.voices.append(self.voiceFeatures(self.record(rc = 3)))
        print("...Finished")
        self._wait(2) 
        
        print("5-second sample- Speak...")
        self.voices.append(self.voiceFeatures(self.record(rc = 5)))
        self._wait(3)
        print("...Finished")
        print("Done!")

    def addVoicesFromPath(self, path, name, add=False):
        for i in os.listdir(path):
            self.voices.append(PostProcess.voiceFeatures(PreProcess.features_extractor(path + i)))
        
        if add:
            self.addGroup()
            self.whois[len(self.groups)] = name

    def addVoice(self, voice, addone=False):
        self.voices.append(PostProcess.voiceFeatures(voice))
        if addone: self.addGroup()

    def generateCluster(self):
        self.cluster = KMeans(
                                n_clusters = self.groups, 
                                init = "k-means++", 
                                max_iter=300, 
                                n_init=1, 
                                verbose=0, 
                                random_state=3425)
        self.cluster.fit(self._train())
        self.generated = True

    def predict(self, voice, thresh:float = -20) -> str:
        if not self.generated: 
            print("First Generate the Clusters")
            return
        vf = self.voiceFeatures(voice)
        score = self.cluster.score(vf)

        if not (score >= thresh): return "Unknown" 
        return self.whois[self.cluster.predict(vf)[0]]

    def predictFromPath(self, path, thresh:float = -20) -> str:
        if not self.generated: 
            print("First Generate the Clusters")
            return
        vf = self.voiceFeatures(self.voiceFeatures(PreProcess.features_extractor(path)))
        score = self.cluster.score(vf)
        
        if not (score >= thresh): return "Unknown"
        return self.whois[self.cluster.predict(vf)[0]]

    def voiceFeatures(self, voice):
        tnsr = tensorflow.convert_to_tensor(voice)[tensorflow.newaxis, ...]
        return np.reshape(PostProcess.vf.predict(tnsr), (PostProcess.vf.layers[-1].output.shape[1]))