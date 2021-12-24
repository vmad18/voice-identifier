# voice-identifier
Identifying a speaker, with their voice, from a couple of samples

# How It Works
A dataset of known voices of varying lengths as well as known random noises are used to train a deep learning model. 
Mel-frequency cepstrum (mfccs) features are extracted from the sounds and train the model. MFCCs describe the "shape" of the audio. 
The output of the model was whether the sound was a voice, and who that voice was, or if it was noise. 

The trained model has an accuracy of: *~90.1%*

The model was now trained to not only discern voices from sounds, but discern voices from other voices. 
With that intuition, the last layer of the model was removed, and the new model outputed a voice feature vector.

Two similar voices predicted through this new model should yield similar feature vectors. 

If one clustered these vectors, any foreign voice could be matched with a cluster of best fit.
Thus, no need to retrain the model to identify a new voice! 

KMean clustering was used in order to cluster the vectors. 
KMean clustering is a form of unsupervised learning that tries to group vectors to the best matched centroid of 'K' centroids (total number of groups). 

# Requirments
- Tensorflow, latest stable
- Sci-Kit Learn, latest stable
- Librosa, latest stable
- PyAudio, latest stable 

# How to Use
Examples can be found in the main.py file.
You can either retrain the model with additional voices or use the already pretrained model.

# Datasets Used

* OpenSLR LibriSpeech-dev-clean (voices): https://www.openslr.org/resources/12/dev-clean.tar.gz
* OpenSLR LibriSpeech-test-clean (voices): https://www.openslr.org/resources/12/test-clean.tar.gz
* Environmental Sound Classification (random noise): https://github.com/karoldvl/ESC-50/archive/master.zip
