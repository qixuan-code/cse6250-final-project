import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText

import collections
import gc 

import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input, concatenate, Activation, Concatenate, LSTM, GRU#merge 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, BatchNormalization, GRU, Convolution1D, LSTM
from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPool1D#, merge

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from keras.utils import np_utils
import tensorflow as tf


from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l2
import warnings
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import AUC, Precision, Recall
warnings.filterwarnings('ignore')
import json
import argparse

def create_dataset(dict_of_ner,emb_type):
    temp_data = []
    for k, v in dict_of_ner.items():
        temp = []
        for embed in v:
            if np.ndim(embed) > 0:  
                temp.append(list(embed))
                
        if emb_type == 'concat':
            mean_embedding = np.mean(temp, axis=0) if temp else np.zeros(200)
        else:
            mean_embedding = np.mean(temp, axis=0) if temp else np.zeros(100)
        temp_data.append(mean_embedding)

    return np.asarray(temp_data)

def make_prediction_multi_avg(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i>=0.5 else 0 for i in probs]
    return probs, y_pred

def save_scores_multi_avg(predictions, probs, ground_truth, 
                          
                          embed_name, problem_type, iteration, hidden_unit_size,
                          
                          sequence_name, type_of_ner):
    
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    
    result_dict = {}    
    result_dict['auc'] = auc
    result_dict['auprc'] = auprc
    result_dict['acc'] = acc
    result_dict['F1'] = F1
    
    result_path = "results/"
    file_name = str(sequence_name)+"-"+str(hidden_unit_size)+"-"+embed_name
    file_name = file_name +"-"+problem_type+"-"+str(iteration)+"-"+type_of_ner+"-avg-.p"
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    print(auc, auprc, acc, F1)

def avg_ner_model(layer_name, number_of_unit, embedding_name):

    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = Input(shape=(24,104))

    input_avg = Input(shape=(input_dimension, ), name = "avg")  
    print("input_avg:",input_avg.shape)
#     x_1 = Dense(256, activation='relu')(input_avg)
#     x_1 = Dropout(0.3)(x_1)
    
    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)

    x = keras.layers.Concatenate()([x, input_avg])

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    preds = Dense(1, activation='sigmoid',use_bias=False,
                         kernel_initializer=GlorotNormal(), 
                  kernel_regularizer=l2(0.01))(x)
    
    
    opt = Adam(lr=0.001, decay = 0.01)
    model = Model(inputs=[sequence_input, input_avg], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[
                  AUC(name='roc_auc', curve='ROC'),  
                  AUC(name='pr_auc', curve='PR'),
                  Precision(name='precision'),
                  Recall(name='recall')
              ])
    
    return model

def mean(a):
    return sum(a) / len(a)

def main(emb_type):
    metrics = {
        'roc_auc': {},
        'val_roc_auc': {},
        'pr_auc': {},
        'val_pr_auc': {},
        'precision': {},
        'val_precision': {},
        'recall': {},
        'val_recall': {}
        }
    type_of_ner = "new"

    x_train_lstm = pd.read_pickle("data/"+type_of_ner+"_x_train.pkl")
    x_dev_lstm = pd.read_pickle("data/"+type_of_ner+"_x_dev.pkl")
    x_test_lstm = pd.read_pickle("data/"+type_of_ner+"_x_test.pkl")

    y_train = pd.read_pickle("data/"+type_of_ner+"_y_train.pkl")
    y_dev = pd.read_pickle("data/"+type_of_ner+"_y_dev.pkl")
    y_test = pd.read_pickle("data/"+type_of_ner+"_y_test.pkl")

    ner_word2vec = pd.read_pickle("data/"+type_of_ner+"_ner_word2vec_limited_dict.pkl")
    ner_fasttext = pd.read_pickle("data/"+type_of_ner+"_ner_fasttext_limited_dict.pkl")
    ner_concat = pd.read_pickle("data/"+type_of_ner+"_ner_combined_limited_dict.pkl")

    train_ids = pd.read_pickle("data/"+type_of_ner+"_train_ids.pkl")
    dev_ids = pd.read_pickle("data/"+type_of_ner+"_dev_ids.pkl")
    test_ids = pd.read_pickle("data/"+type_of_ner+"_test_ids.pkl")
    
    #embedding_types = ['word2vec']
    embedding_types = [emb_type]
    if emb_type == 'word2vec':
        embedding_dict = [ner_word2vec]
    elif emb_type == 'fasttext':
        embedding_dict = [ner_fasttext]
    else:
        embedding_dict = [ner_concat]
        
        
    target_problems = ['mort_hosp']

    num_epoch = 100
    model_patience = 5
    monitor_criteria = 'val_loss'
    batch_size = 64
    iter_num = 2
    unit_sizes = [128]

    layers = ["GRU"]

    for each_layer in layers:
        print ("Layer: ", each_layer)
        for each_unit_size in unit_sizes:
            print ("Hidden unit: ", each_unit_size)

            for embed_dict, embed_name in zip(embedding_dict, embedding_types):    
                print ("Embedding: ", embed_name)
                print("=============================")

                temp_train_ner = dict((k, ner_word2vec[k]) for k in train_ids)
                temp_dev_ner = dict((k, ner_word2vec[k]) for k in dev_ids)
                temp_test_ner = dict((k, ner_word2vec[k]) for k in test_ids)

                x_train_ner = create_dataset(temp_train_ner,emb_type)
                x_dev_ner = create_dataset(temp_dev_ner,emb_type)
                x_test_ner = create_dataset(temp_test_ner,emb_type)


                for iteration in range(1, iter_num):
                    print ("Iteration number: ", iteration)

                    for each_problem in target_problems:
                        print ("Problem type: ", each_problem)
                        print ("__________________")

                        early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, patience=model_patience)
                        best_model_name = "avg-"+str(embed_name)+"-"+str(each_problem)+"-"+"best_model.hdf5"
                        checkpoint = ModelCheckpoint(best_model_name, monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min', period=1)


                        callbacks = [early_stopping_monitor, checkpoint]

                        model = avg_ner_model(each_layer, each_unit_size, embed_name)

                        history = model.fit([x_train_lstm, x_train_ner], y_train[each_problem], epochs=num_epoch, verbose=1, 
                                  validation_data=([x_dev_lstm, x_dev_ner], y_dev[each_problem]), callbacks=callbacks, 
                                  batch_size=batch_size )

                        model.load_weights(best_model_name)

                        probs, predictions = make_prediction_multi_avg(model, [x_test_lstm, x_test_ner])

                        save_scores_multi_avg(predictions, probs, y_test[each_problem], 
                                   embed_name, each_problem, iteration, each_unit_size, 
                                   each_layer, type_of_ner)
                        history_data = history.history
                        for metric in metrics:
                            if metric in history_data:
                                if iteration not in metrics[metric]:
                                    metrics[metric][iteration] = []
                                metrics[metric][iteration].extend(history_data[metric])

         
                        #clear_session()
                        gc.collect()
                        
    # Convert the metrics dictionary to a JSON string
    metrics_json = json.dumps(metrics)

    # Save the JSON s/tring to a file
    with open(f'results/section8_metrics{emb_type}.json', 'w') as f:
        f.write(metrics_json)                       
if __name__ == "__main__":
    #main()
    parser = argparse.ArgumentParser(description="Train a model with specified embedding type.")
    
    parser.add_argument('--embedding_type', type=str, default='word2vec',
                        choices=['word2vec', 'fasttext', 'concat'],
                        help='Type of word embedding to use')

    args = parser.parse_args()

    main(args.embedding_type)
