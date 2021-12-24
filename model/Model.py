import tensorflow
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics


class VoiceRecognition:

    def __init__(self, ep, bs, feats, l, xt, yt, xv, yv, sp, m = Sequential()):
        self.model = m
        self.voicefeat = None
        self.epochs = ep
        self.batchs = bs
        self.n_feats = feats 
        self.labels = l
        self.x_train = xt
        self.y_train = yt
        self.x_val =  xv
        self.y_val = yv
        self.save_path = sp

    def generate(self):
        self.model.add(BatchNormalization(input_shape=self.n_feats[0].shape,))

        self.forward(100, "relu") 
        self.forward(200, "relu")
        self.forward(250, "sigmoid")
        self.forward(200, "tanh")

        self.model.add(Dense(self.labels))
        self.model.add(Activation("softmax"))

        self.model.compile(
                        loss = "categorical_crossentropy", 
                        metrics=["accuracy"], 
                        optimizer="adam")
        self.model.summary()

    def forward(self, nodes, activ):
        self.model.add(Dense(nodes)) 
        self.model.add(Activation(activ))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())

    def train(self):
        model_check = ModelCheckpoint(
                            filepath=self.save_path, 
                            save_best_only=True,
                            verbose = 1)
        self.model.fit(
            self.x_train, self.y_train, 
            batch_size=self.batchs, epochs=self.epochs, 
            validation_data=(self.x_val, self.y_val), 
            callbacks=[model_check], 
            verbose=1)

    def eval(self):
        return self.model.evaluate(
                                    self.x_val, 
                                    self.y_val, 
                                    verbose=0)

    def predict(self, voice):
        tnsr = tensorflow.convert_to_tensor(voice)[tensorflow.newaxis, ...]
        return np.argmax(self.model.predict(tnsr))

    def load_model(self, model):
        self.model = model

    '''
    Generate Feature Vector Model
    '''
    def generatevf(self):
        self.voicefeat = Model(
            inputs = self.model.input, 
            outputs = self.model.layers[-3].output)

        self.voicefeat.compile(
            loss = "categorical_crossentropy", 
            metrics = ["accuracy"], 
            optimizer = "adam")
        return self.voicefeat