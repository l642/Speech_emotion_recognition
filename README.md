# Speech_emotion_recognition
Speech Emotion Recognition (SER), is the act of attempting to recognize human emotion and affective states from speech.In this Python based project, we have used the libraries librosa, soundfile, and sklearn (among others) to build a model using an MLPClassifier. This will be able to recognize emotion from sound files. We have loaded the data, extracted the audio features (mfcc, Stft,Melspectogram) from it, then split the dataset into training and testing sets. Then, we’ll initialize an MLPClassifier and train the model. Finally, we’ll calculate the accuracy of our model.
### Dataset
for this Python based project, we have used use the RAVDESS dataset; this is the Ryerson Audio-Visual Database of Emotional Speech and Song dataset.

### Steps for speech emotion recognition
```
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
```

