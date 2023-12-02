import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText
# import glove
# from glove import Corpus

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


from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')
import json
import argparse

def mean(a):
    return sum(a) / len(a)

def make_prediction_cnn(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i>=0.5 else 0 for i in probs]
    return probs, y_pred

def save_scores_cnn(predictions, probs, ground_truth, embed_name, problem_type, iteration, hidden_unit_size,sequence_name, type_of_ner):
    
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    
    result_dict = {}    
    result_dict['auc'] = auc
    result_dict['auprc'] = auprc
    result_dict['acc'] = acc
    result_dict['F1'] = F1

    result_path = "results/cnn/"
    file_name = str(sequence_name)+"-"+str(hidden_unit_size)+"-"+embed_name
    file_name = file_name +"-"+problem_type+"-"+str(iteration)+"-"+type_of_ner+"-cnn-.p"
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    print(auc, auprc, acc, F1)

def print_scores_cnn(predictions, probs, ground_truth, model_name, problem_type, iteration, hidden_unit_size):
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    
    print ("AUC: ", auc, "AUPRC: ", auprc, "F1: ", F1)

def get_subvector_data(size, embed_name, data):
    if embed_name == "concat":
        vector_size = 200
    else:
        vector_size = 100

    x_data = {}
    for k, v in data.items():
        
        v_filtered = [x for x in v if x.size == vector_size]
        if v_filtered:
            v_2d = np.vstack(v_filtered)
      
            if v_2d.shape[0] < size:
                padding = np.zeros((size - v_2d.shape[0], vector_size))
                v_2d = np.vstack((v_2d, padding))
            else:
                v_2d = v_2d[:size]
        else:
            v_2d =  np.zeros((size, vector_size))
        x_data[k] = v_2d
    return x_data


def proposedmodel(layer_name, number_of_unit, embedding_name, ner_limit, num_filter):
    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = Input(shape=(24,104))

    input_img = Input(shape=(ner_limit, input_dimension), name = "cnn_input")

    convs = []
    filter_sizes = [2,3,4]



    text_conv1d = Conv1D(filters=num_filter, kernel_size=3, 
                 padding = 'valid', strides = 1, dilation_rate=1, activation='relu', 
                         kernel_initializer=tf.keras.initializers.GlorotUniform() )(input_img)
    
    text_conv1d = Conv1D(filters=num_filter*2, kernel_size=3, 
                 padding = 'valid', strides = 1, dilation_rate=1, activation='relu',
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(text_conv1d)   
    
    text_conv1d = Conv1D(filters=num_filter*3, kernel_size=3, 
                 padding = 'valid', strides = 1, dilation_rate=1, activation='relu',
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(text_conv1d)   

    
    #concat_conv = keras.layers.Concatenate()([text_conv1d, text_conv1d_2, text_conv1d_3])
    text_embeddings = GlobalMaxPooling1D()(text_conv1d)
    #text_embeddings = Dense(128, activation="relu")(text_embeddings)
    
    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)

    concatenated = concatenate([x, text_embeddings], axis=1)
    concatenated = Dense(512, activation='relu')(concatenated)
    concatenated = Dropout(0.2)(concatenated)

    
    preds = Dense(1, activation='sigmoid',use_bias=False,
                         kernel_initializer=GlorotNormal(), 
                  kernel_regularizer=l2(0.01))(concatenated)

    
    opt = Adam(lr=1e-3, decay = 0.01)
    

    model = Model(inputs=[sequence_input, input_img], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[
                  AUC(name='roc_auc', curve='ROC'), 
                  AUC(name='pr_auc', curve='PR'),
                  Precision(name='precision'),
                  Recall(name='recall')
              ])
    
    return model

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
    
    embedding_types = [emb_type]
    if emb_type == 'word2vec':
        embedding_dict = [ner_word2vec]
    elif emb_type == 'fasttext':
        embedding_dict = [ner_fasttext]
    else:
        embedding_dict = [ner_concat]
        
    # embedding_types = ['word2vec']
    # embedding_dict = [ner_word2vec]

    target_problems = ['mort_hosp']

    num_epoch = 100
    model_patience = 5
    monitor_criteria = 'val_loss'
    batch_size = 64

    filter_number = 32
    ner_representation_limit = 64
    activation_func = "relu"

    sequence_model = "GRU"
    sequence_hidden_unit = 256

    maxiter = 11
    for embed_dict, embed_name in zip(embedding_dict, embedding_types):    
        print ("Embedding: ", embed_name)
        print("=============================")

        temp_train_ner = dict((k, embed_dict[k]) for k in train_ids)
        tem_dev_ner = dict((k, embed_dict[k]) for k in dev_ids)
        temp_test_ner = dict((k, embed_dict[k]) for k in test_ids)

        x_train_dict = {}
        x_dev_dict = {}
        x_test_dict = {}

        x_train_dict = get_subvector_data(ner_representation_limit, embed_name, temp_train_ner)
        x_dev_dict = get_subvector_data(ner_representation_limit, embed_name, tem_dev_ner)
        x_test_dict = get_subvector_data(ner_representation_limit, embed_name, temp_test_ner)

        x_train_dict_sorted = collections.OrderedDict(sorted(x_train_dict.items()))
        x_dev_dict_sorted = collections.OrderedDict(sorted(x_dev_dict.items()))
        x_test_dict_sorted = collections.OrderedDict(sorted(x_test_dict.items()))

        x_train_ner = np.stack(list(x_train_dict_sorted.values()), axis=0)
        x_dev_ner = np.stack(list(x_dev_dict_sorted.values()), axis=0)
        x_test_ner = np.stack(list(x_test_dict_sorted.values()), axis=0)
        for iteration in range(1,maxiter):
            print ("Iteration number: ", iteration)

            for each_problem in target_problems:
                print ("Problem type: ", each_problem)
                print ("__________________")


                early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, patience=model_patience)

                best_model_name = str(ner_representation_limit)+"-basiccnn1d-"+str(embed_name)+"-"+str(each_problem)+"-"+"best_model.hdf5"

                checkpoint = ModelCheckpoint(best_model_name, monitor=monitor_criteria, verbose=1,
                    save_best_only=True, mode='min')

                reduce_lr = ReduceLROnPlateau(monitor=monitor_criteria, factor=0.2,
                                  patience=2, min_lr=0.00001, epsilon=1e-4, mode='min')


                callbacks = [early_stopping_monitor, checkpoint, reduce_lr]

            
                model = proposedmodel(sequence_model, sequence_hidden_unit, 
                                   embed_name, ner_representation_limit,filter_number)
                history= model.fit([x_train_lstm, x_train_ner], y_train[each_problem], epochs=num_epoch, verbose=1, 
                          validation_data=([x_dev_lstm, x_dev_ner], y_dev[each_problem]), callbacks=callbacks, batch_size=batch_size)


                probs, predictions = make_prediction_cnn(model, [x_test_lstm, x_test_ner])
                print_scores_cnn(predictions, probs, y_test[each_problem], embed_name, each_problem, iteration, sequence_hidden_unit)

                model.load_weights(best_model_name)

                probs, predictions = make_prediction_cnn(model, [x_test_lstm, x_test_ner])
                save_scores_cnn(predictions, probs, y_test[each_problem], embed_name, each_problem, iteration,
                                sequence_hidden_unit, sequence_model, type_of_ner)
                del model
                clear_session()
                gc.collect()
                
    metrics_json = json.dumps(metrics)

    with open(f'results/section9_metrics_{emb_type}.json', 'w') as f:
        f.write(metrics_json) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified embedding type.")
    
    parser.add_argument('--embedding_type', type=str, default='word2vec',
                        choices=['word2vec', 'fasttext', 'concat'],
                        help='Type of word embedding to use')

    args = parser.parse_args()

    main(args.embedding_type)
