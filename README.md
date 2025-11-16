# Drum Sample Classifier (Kick / Snare / Clap)

This project builds a small convolutional neural network to classify drum samples into three categories: kick, snare, and clap. It is designed as a practical machine-learning tool for electronic music production workflows, where organizing large sample libraries can be slow and repetitive.

The model is trained using a folder of labelled audio samples. Each audio file is converted into a log-mel spectrogram and used as input to a lightweight CNN. After training, the model can automatically classify new samples and sort them into folders.

### Features
- Converts audio files into log-mel spectrograms using torchaudio.
- CNN classifier implemented in PyTorch.
- Automatic folder-based sample sorting after inference.
- Designed to work with arbitrary personal sample libraries.
- Simple structure for easy extension (more classes, bigger model, augmentation, etc.).

### How to Use
1. Place your drum samples inside a `data/` folder with subfolders:
   - `kick/`
   - `snare/`
   - `clap/`
2. Run the training script to create `drum_cnn.pth`.
3. Place new, unlabeled samples inside `unsorted/`.
4. Run the classification script to automatically sort them.

### Requirements
Python 3.10 or later  
PyTorch  
torchaudio  
matplotlib  
librosa  
(any additional libraries used in your scripts)

### Future Improvements
- More robust dataset and additional drum types.
- Data augmentation for generalization.
- Improved UI for drag-and-drop sample sorting.
- Exporting the trained model into a JUCE plugin for DAWs.

This project is part of my audio-focused machine learning work for program applications and portfolio development.
