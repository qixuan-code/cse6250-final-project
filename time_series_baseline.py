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
from keras.layers import Flatten, Dense, Dropout, Input, concatenate, Activation, Concatenate, LSTM, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, BatchNormalization, GRU, Convolution1D, LSTM
from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPool1D

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from keras.utils import np_utils
#from keras.backend.tensorflow_backend import set_session, clear_session, get_session
from keras.backend import set_session, clear_session, get_session

import tensorflow as tf
from tensorflow.keras.models import Model

from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from tensorflow.keras.metrics import AUC, Precision, Recall
import warnings
from tensorflow.keras import backend as K
import gc
warnings.filterwarnings('ignore')
import json

# Reset Keras Session
def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    gc.collect() # if it's done something you should see a number being outputted

def make_prediction_timeseries(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i>=0.5 else 0 for i in probs]
    return probs, y_pred

def save_scores_timeseries(predictions, probs, ground_truth, model_name, 
                problem_type, iteration, hidden_unit_size, type_of_ner):
    
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    
    
    result_dict = {}    
    result_dict['auc'] = auc
    result_dict['auprc'] = auprc
    result_dict['acc'] = acc
    result_dict['F1'] = F1

        
    file_name = str(hidden_unit_size)+"-"+model_name+"-"+problem_type+"-"+str(iteration)+"-"+type_of_ner+".p"
    
    result_path = "results/"
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    print(auc, auprc, acc, F1)

def timeseries_model(layer_name, number_of_unit):
    K.clear_session()
    
    sequence_input = Input(shape=(24,104),  name = "timeseries_input")
    
    if layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)
    else:
        x = GRU(number_of_unit)(sequence_input)
    
    logits_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    sigmoid_pred = Dense(1, activation='sigmoid',use_bias=False,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                  kernel_regularizer=logits_regularizer)(x)
    
    
    model = Model(inputs=sequence_input, outputs=sigmoid_pred)
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def timeseries_model(layer_name, number_of_unit):
    # Clear session, useful to prevent clutter from old models/layers.
    tf.keras.backend.clear_session()
    
    sequence_input = Input(shape=(24,104),  name = "timeseries_input")
    
    if layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)
    else:
        x = GRU(number_of_unit)(sequence_input)
    
    # Replacing tf.contrib.layers with tf.keras alternatives
    logits_regularizer = tf.keras.regularizers.l2(l=0.01)
    sigmoid_pred = Dense(
        1, 
        activation='sigmoid',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.GlorotUniform(), 
        kernel_regularizer=logits_regularizer
    )(x)
    
    model = Model(inputs=sequence_input, outputs=sigmoid_pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
                  AUC(name='roc_auc', curve='ROC'),  # ROC AUC
                  AUC(name='pr_auc', curve='PR'),
                  Precision(name='precision'),# Precision-Recall AUC
                  Recall(name='recall')
              ])
    
    return model


def main():
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
    epoch_num = 100
    model_patience = 3
    monitor_criteria = 'val_loss'
    batch_size = 128

    unit_sizes = [128]
    iter_num = 5
    target_problems = ['mort_hosp']
    layers = ["GRU"]
    
    for each_layer in layers:
        print("Layer: ", each_layer)
        for each_unit_size in unit_sizes:
            print("Hidden unit: ", each_unit_size)
            for iteration in range(1, iter_num):
                print("Iteration number: ", iteration)
                print("=============================")

                for each_problem in target_problems:
                    print ("Problem type: ", each_problem)
                    print ("__________________")

                    early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, patience=model_patience)
                    best_model_name = str(each_layer)+"-"+str(each_unit_size)+"-"+str(each_problem)+"-"+"best_model.hdf5"
                    checkpoint = ModelCheckpoint(best_model_name, monitor='val_loss', verbose=1,
                        save_best_only=True, mode='min', period=1)

                    callbacks = [early_stopping_monitor, checkpoint]

                    model = timeseries_model(each_layer, each_unit_size)  # Assuming this function is defined elsewhere and creates an appropriate tf.keras model
                    history = model.fit(x_train_lstm, y_train[each_problem], epochs=epoch_num, verbose=1, 
                              validation_data=(x_dev_lstm, y_dev[each_problem]), callbacks=callbacks, batch_size= batch_size)

                    model.load_weights(best_model_name)

                    probs, predictions = make_prediction_timeseries(model, x_test_lstm) 
                    save_scores_timeseries(predictions, probs, y_test[each_problem].values,str(each_layer),
                                           each_problem, iteration, each_unit_size,type_of_ner) 
                    
                    history_data = history.history
                    for metric in metrics:
                        if metric in history_data:
                            if iteration not in metrics[metric]:
                                metrics[metric][iteration] = []
                            metrics[metric][iteration].extend(history_data[metric])
                    reset_keras(model)
                    gc.collect()
   
    # Convert the metrics dictionary to a JSON string
    metrics_json = json.dumps(metrics)

    # Save the JSON s/tring to a file
    with open('results/section7_metrics.json', 'w') as f:
        f.write(metrics_json)

if __name__ == "__main__":
    main()
