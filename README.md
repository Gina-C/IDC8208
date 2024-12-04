# IDC8208 Project: Speech Denoising and Speaker Characteristic Inferencing 
This repository contains a Colab notebook, which is a pipeline designed for speech denoising and classification of speaker attributes. The pipeline integrates standardisation of audio, AI-based denoising model, assess quality of denoising using PESQ, and classification of speaker attributes such as gender, age, and height. 

## Features
1. **Download audio file from YouTube**: Saving YouTube videos as audio files (.wav)
2. **Load and standardise audio format**: Standardizes sample rate and mono audio for any audio input file
3. **Denoise audio & Calculate PESQ**: Using AI-based denoising technique to improve audio quality, then use PESQ (Perceptual Evaluation of Speech Quality) metric to assess quality
4. **Speaker Diarization**: Segments audio files to label a speaker for each audio segment file
5. **Classification Model for Gender, Age and Height**: Implement and combine 2 pre-trained models for prediction of:
    - Gender
    - Age
    - Height
   
## Dependencies
You can install the required libraries in this notebook using:
```
!pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
!pip install pyannote.audio
!pip install librosa soundfile speechbrain yt-dlp pydub pypesq
```

You can import the required libraries in this notebook using:
```
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import librosa
import torch
import yt_dlp
import os
import soundfile as sf
import torchaudio
from speechbrain.inference.enhancement import WaveformEnhancement
from pypesq import pesq
from pyannote.audio import Pipeline
import csv
import pandas as pd
from pydub import AudioSegment
import torchaudio.transforms as T
```
## Setup and Usage
1. **Open the Notebook**: Use the following link to access the Colab notebook:  
   [Speech Processing Pipeline Notebook](https://colab.research.google.com/drive/1yntv5XsZQOrnoYIa05q3Z1lluMtlYsCr?usp=sharing)
2. **Duplicate the Notebook**: Save a copy of the notebook in your Google Drive. 

3. **Edit the Fields "## To change"**: Replace all the variables and code chunk that have the "## To change" label to the data of your choice. 

4. **Run the Pipeline**: Execute all cells until and including "Pipeline(Full)" section. There are 2 parts of the pipeline.
   
5. **Use the Pipeline**: Execute the Test Case cells sequentially until pipeline part 1. From the result, mapped the Original Speaker label to Predicted Speaker label, before creating the actual_class that have true labels for the Predicted Speaker. Run pipeline part 2 to visualise results.

6. **Results**: The notebook provides detailed visualizations and tabular outputs for:
   - PESQ Score
   - Results sorted by Speaker
   - Speaker and Timestamp from diarization
   - Classification results for Age and Height in Boxplot
   - Classification results for Gender in Confusion Matrix
  
## File Structure
- **Notebook**: Contains the full pipeline for speech denoising and classification of speaker attributes.

## Future Improvements
- Upgraded AI-denoising models
- Impproved model for speaker diarization
- Improved models for classification. 

## License

This project is licensed under the [MIT License](LICENSE).
