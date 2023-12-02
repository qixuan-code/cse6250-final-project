# cse6250-final-project

## Data acquisition:
##### note: files from MIMIC can only be accessed with proper training
- step 1: Run MIMIC-Extract Pipeline as explained in https://github.com/MLforHealth/MIMIC_Extract. or you can get the output file all_hourly_data.h5 from [here](https://console.cloud.google.com/storage/browser/mimic_extract;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). Download it under data folder

- step 2: Copy the ADMISSIONS.csv, NOTEEVENTS.csv, ICUSTAYS.csv files into data folder.

- step 3: Download Pre-trained Word2Vec & FastText embeddings: https://github.com/kexinhuang12345/clinicalBERT
download it under embeddings folder

## Data Preprocessing (Steps 01 to 06):
##### note: those data preprocess steps are strictly followed, which is provided by author to ensure reproducability
- step 01-Extract-Timseries-Features.ipnyb to extract first 24 hours timeseries features from MIMIC-Extract raw data.

- step 02-Select-SubClinicalNotes.ipynb to select subnotes based on criteria from all MIMIC-III Notes.

- step 03-Prprocess-Clinical-Notes.ipnyb to prepocessing notes.

- step 04-Apply-med7-on-Clinical-Notes.ipynb to extract medical entities.

- step 05-Represent-Entities-With-Different-Embeddings.ipynb to convert medical entities into word representations.

- step 06-Create-Timeseries-Data.ipynb to prepare the timeseries data to fed through GRU / LSTM.

## Model Training:

##### note:Execute the following Python scripts in sequence to train different models. Model training part is modified from literature code,the structure of model follows the description of the literature, some packages are upgraded and debugged. 
- time_series_baseline.py: This script trains a baseline model using Long Short-Term Memory (LSTM) neural networks. You can run by "python time_series_baseline.py"
- multimodal_baseline.py: This script is for training a baseline multimodal model. You can run by "python multimodal_baseline.py --embedding_type concat";"python multimodal_baseline.py --embedding_type word2vec" ;"python multimodal_baseline.py --embedding_type fasttext" 
- proposed_model.py: Run this script to train the proposed CNN model. You can run by "python proposed_model.py --embedding_type concat";"python proposed_model.py --embedding_type word2vec" ;"python proposed_model.py --embedding_type fasttext" 
- after training, it will save the model training metrics to results as json file.

## Model Performance Analysis:
##### note:This part generates the data and figure for final report, which is written by us.
- metrics_analysis.ipynb: this file reads the model training metrics, conduct post-analysis and compare the metrics of different models with various embeddings. 