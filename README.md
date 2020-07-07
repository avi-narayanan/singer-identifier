# Problem Statement
As a singer, I tend to record a lot of songs (bhajans specificaly, [songs like these](https://www.youtube.com/watch?v=0w6xY3CN-j8)) on my phone, sang by other singers and myself alike. The result is a jumble of songs with generic names like "My Recording 67.wav". My loved ones would often ask me to send songs that I have sung and I found it very difficult to find anything in this mess. 
I took the opportunity to solve this problem with machine learning. 

# Methodology

## TL;DR
I used voice recordings that I had gathered on my phone over the last year (total of ~350 bhajans, ~80 were sang by me, ~20 different singers). I developed a [simple html/js tool](https://github.com/avi-narayanan/singer-identifier/tree/master/Song%20Annotator) that would help annotate the songs, shared subsets of the data to close friends and family members and within a few weeks I had a usable dataset. I then converted the dataset into 4 second spectrograms that could be fed into a deep neural net based on VGGish model. 

I used two different methods to identify my voice in a given snipet of audio
- Generalizable model - <br />Used a siamese network to train a model that generates a "fingerprint" of a given singer. A new audio sample is compared to the fingerprint using a distance metric and is classified as my voice if the distance is within a defined threshold
- Non Generalizable model 
  - Binary Classifier - Trained a model that predicts whether a given spectrogram is my voice or not
  - Multi-class Classifier - Trained a model that that predicts whether a given spectrogram is one or many different singers present in the dataset

### Results
- Non generalizable models performed much better in identifying my voice, >99% accuracy and recall for both binary and multi class classifier
- The siamese network performed very well at the task of distinguishing between two artists (>90% accuracy on validation data). This however did not directly translate into stellar performance in the one-shot learning task. Using an average of "fingerprints" generated for spectrograms as the my voice's fingerprint, I was able to identify my songs with a ~70% accuracy. 

For details, checkout the [blog]().

# Future Work
- Add more variety to the dataset and retrain (more female singers, more songs without any percussion and supporting instruments)
- Evaluate different methods of generating a fingerprint for a singer
- Deploy model as a consumable API

# References
* Keras Implementations of VGGish : https://github.com/DTaoo/VGGish
* Gemmeke, J. et. al., [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html), ICASSP 2017
* Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://research.google/pubs/pub45611/), ICASSP 2017
* [One Shot Learning with Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
