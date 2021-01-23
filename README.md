# Speech_emotion_recognition
Speech Emotion Recognition (SER), is the act of attempting to recognize human emotion and affective states from speech.In this Python based project, we have used the libraries librosa, soundfile, and sklearn (among others) to build a model using an MLPClassifier. This will be able to recognize emotion from sound files. We have loaded the data, extracted the audio features (mfcc, Stft,Melspectogram) from it, then split the dataset into training and testing sets. Then, we’ll initialize an MLPClassifier and train the model. Finally, we’ll calculate the accuracy of our model.
### Dataset
for this Python based project, we have used use the RAVDESS dataset; this is the Ryerson Audio-Visual Database of Emotional Speech and Song dataset.

### Steps for speech emotion recognition
### 1. Import the important libraries
```
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
```

### 2. Feature Extraction
 Define a function extract_feature to extract the mfcc, chroma, and mel features from a sound file. This function takes 4 parameters- the file name and three Boolean parameters for the three features:
 
  mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
  
  chroma: Pertains to the 12 different pitch classes
  
  mel: Mel Spectrogram Frequency
 Open the sound file with soundfile.SoundFile using with-as so it’s automatically closed once we’re done. Read from it and call it X. Also, get the sample rate. If chroma is True, get the Short-Time Fourier Transform of X.

Let result be an empty numpy array. Now, for each feature of the three, if it exists, make a call to the corresponding function from librosa.feature (eg- librosa.feature.mfcc for mfcc), and get the mean value. Call the function hstack() from numpy with result and the feature value, and store this in result. hstack() stacks arrays in sequence horizontally (in a columnar fashion). Then, return the result.
