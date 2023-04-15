import concurrent
import copy
import ctypes
import itertools
import math
import sys
from concurrent.futures import wait
from statistics import mean

import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor, Perceptron, PassiveAggressiveClassifier, \
    SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import OrderedDict
import util
from keras.layers import Flatten, Dense, Reshape, MaxPooling1D, Bidirectional
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn import preprocessing
import feature
from itertools import combinations
from sklearn.feature_extraction.text import TfidfTransformer
import multiprocessing as mp
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
import pandas as pd
from sklearn.utils import compute_class_weight
from keras.layers import Conv1D, Conv2D
from keras.models import Model
from keras.utils.data_utils import *
from autoencoder import *
from keras.layers import Dense, Input
from keras.layers import Flatten
from multiprocessing.managers import BaseManager
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.model_selection import train_test_split
import threading
import platform

global sample_size
global sampling

os.environ["TOKENIZERS_PARALLELISM"] = "false"

filename = 'DDIdataset'
k=10000
num_folds=10

import ctypes

class DL:

    def __init__(self, n_iter=100):
        self.n_iter = n_iter
        self.batch_size = None
        self.model = None

    def fit(self, train_x, train_y):
        if sampling == 'over_sampling':
            batch_size = min(int(train_x.shape[0] / 2), 256)
        elif sampling == 'no_sampling':
            batch_size = min(int(train_x.shape[0] / 2), 128)
        else:
            batch_size = min(int(train_x.shape[0] / 2), 64)
        if train_x.shape[0] < batch_size:
            batch_size = int(train_x.shape[0]/2)
        num_batches = int(train_x.shape[0]/batch_size)
        train_x = train_x[0:num_batches * batch_size, :].reshape(num_batches, batch_size, train_x.shape[1], train_x.shape[2])
        train_y = train_y[0:num_batches * batch_size].reshape(num_batches, batch_size)
        input_layer, output_layer = self.get_input_output_layer(train_x, train_y)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        #early_stopping_callback = EarlyStopping(monitor='accuracy', patience=10, mode='min', restore_best_weights=True)
        #model.fit(train_x, train_y, callbacks=[early_stopping_callback], steps_per_epoch=num_batches, verbose=0, epochs=self.n_iter, shuffle=False)
        model.fit(train_x, train_y, steps_per_epoch=num_batches, verbose=0, epochs=self.n_iter, shuffle=False)
        self.batch_size = batch_size
        self.model = model

    def predict(self, X_test):
        num_batches = int(X_test.shape[0] / self.batch_size)
        X_test = X_test[0:num_batches * self.batch_size, :].reshape(num_batches, self.batch_size, X_test.shape[1], X_test.shape[2])
        Y_pred = self.model.predict(X_test, steps=num_batches)
        Y_pred_new = list()
        for pred in Y_pred:
            index = int(np.where(pred == np.amax(pred))[0])
            Y_pred_new.append(index)
        return Y_pred_new

    def predict_proba(self, X_test):
        num_batches = int(X_test.shape[0] / self.batch_size)
        X_test = X_test[0:num_batches * self.batch_size, :].reshape(num_batches, self.batch_size, X_test.shape[1], X_test.shape[2])
        Y_pred = self.model.predict(X_test, steps=num_batches)
        return Y_pred

class DeepConvFFNN(DL):
    def get_input_output_layer(self, train_x, train_y):

        input_layer = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)
        bn2 = tf.reshape(bn2, [bn2.shape[1], bn2.shape[2], bn2.shape[3]])
        #conv3 = Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', padding='same')(bn2)
        #bn3 = BatchNormalization()(conv3)
        #bn3 = tf.reshape(bn3, [bn3.shape[1], bn3.shape[2], bn3.shape[3]])
        #flat = Flatten()(bn3)
        flat = Flatten()(bn2)
        output_layer = Dense(len(np.unique(train_y)), activation='softmax')(flat)
        return input_layer, output_layer

class DeepConvStackedLSTM(DL):
    def get_input_output_layer(self, train_x, train_y):

        input_layer = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)
        bn2 = tf.reshape(bn2, [bn2.shape[1], bn2.shape[2], bn2.shape[3]])
        #conv3 = Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', padding='same')(bn2)
        #bn3 = BatchNormalization()(conv3)
        #bn3 = tf.reshape(bn3, [bn3.shape[1], bn3.shape[2], bn3.shape[3]])
        #lstm1 = LSTM(128, return_sequences=True)(bn3)
        lstm1 = LSTM(128, return_sequences=True)(bn2)
        do1 = Dropout(0.5)(lstm1)
        lstm2 = LSTM(64)(do1)
        do2 = Dropout(0.2)(lstm2)
        flat = Flatten()(do2)
        output_layer = Dense(len(np.unique(train_y)), activation='softmax')(flat)
        return input_layer, output_layer

class DeepConvBiLSTM(DL):
    def get_input_output_layer(self, train_x, train_y):

        input_layer = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)
        bn2 = tf.reshape(bn2, [bn2.shape[1], bn2.shape[2], bn2.shape[3]])
        #conv3 = Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', padding='same')(bn2)
        #bn3 = BatchNormalization()(conv3)
        #bn3 = tf.reshape(bn3, [bn3.shape[1], bn3.shape[2], bn3.shape[3]])
        #lstm1 = Bidirectional(LSTM(128, return_sequences=True))(bn3)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(bn2)
        do = Dropout(0.5)(lstm1)
        flat = Flatten()(do)
        output_layer = Dense(len(np.unique(train_y)), activation='softmax')(flat)
        return input_layer, output_layer

if __name__ == '__main__':
    global sampling
    try:
        sample_size=float(sys.argv[1])
        _type=int(sys.argv[2])
        num_threads = int(sys.argv[3])
        min_components = int(sys.argv[4])
        sampling = sys.argv[5]
    except:
        print('sample_size or num_threads or feature_type or min_components or sampling missing!!!')
        sys.exit(1)

    df = pd.read_csv(filename+'.csv')
    df = df.sample(frac=sample_size, replace=True, random_state=42).reset_index(drop=True)
    df['normalized_sentence'] = df['sentence'].apply(lambda x: util.review_cleaning(x))
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    df = df[['normalized_sentence', 'drug1', 'drug2', 'label', 'type']]

    smiles_feature = feature.Smiles("smiles", k, 1, num_threads)
    biobert_feature = feature.BioBert("biobert", k, 1, num_threads)
    rdf2vec_feature = feature.RDF2VecFeature("rdf2vec", k, 1, num_threads)
    word_feature = feature.WordFeature("freq_words", k, 0, num_threads)
    word_pair_feature = feature.WordPairFeature("freq_phrases", k, 0, num_threads)
    syntactic_grammar_feature = feature.SyntacticGrammerFeature("freq_syntactic_trios", k, 0, num_threads)
    features = [
        smiles_feature,
        biobert_feature,
        rdf2vec_feature,
        word_feature,
        word_pair_feature,
        syntactic_grammar_feature,

    ]
    all_feature_combos = list()
    for i in range(1, len(features) + 1):
        all_feature_combos += list(combinations(features, i))
    feature_combos = list()
    for feature_combo in all_feature_combos:
        if len(set([feature.type for feature in feature_combo])) == 1:
            feature_combos.append(feature_combo)
    feature_combos.sort(key=len)
    for i, feature_combo in enumerate(feature_combos):
        if len(feature_combo) == 1 and feature_combo[0].type == _type :
            gc.collect()
            feature_matrix = feature_combo[0].create_df(df)
            feature_matrix = np.array([value for key, value in sorted(feature_matrix.items())])
            if _type==0:
                feature_matrix = TfidfTransformer().fit_transform(np.asarray(feature_matrix)).toarray()
            num_batches = math.ceil(10 * sample_size)
            num_components = min(feature_matrix.shape[1], int(feature_matrix.shape[0] / num_batches), min_components)
            print('n_components: ', num_components)
            incremental_pca = IncrementalPCA(n_components=num_components)
            for batch in np.array_split(feature_matrix, num_batches):
                incremental_pca.fit(batch)
            feature_matrix = incremental_pca.transform(feature_matrix).tolist()
            df[feature_combo[0].name] = feature_matrix
            feature_size = len(feature_matrix[0])
            feature_combo[0].set_feature_size(feature_size)

    feature_combos.reverse()

    feature_list = list()
    scores = OrderedDict()
    scores2 = OrderedDict()

    if sampling == 'under_sampling':
        _df = util.undersample(df)
    elif sampling == 'over_sampling':
        _df = util.oversample(df)
    else:
        _df = df

    print('dataset stats:')
    print('number of instances', len(_df))
    print('number of negative DDIs', len(_df.query("label == 0")))
    print('number of positive DDIs', len(_df.query("label == 1")))
    print('number of Advise DDIs', len(_df.query('type == "advise"')))
    print('number of Effect DDIs', len(_df.query('type == "effect"')))
    print('number of Mechanism DDIs', len(_df.query('type == "mechanism"')))
    print('number of Int DDIs', len(_df.query('type == "int"')))

    df['type'] = le.fit_transform(df['type'])

    _df2 = _df.query("label == 1")
    train_df, test_df = train_test_split(_df, test_size=0.4, random_state=42)
    train_df2, test_df2 = train_test_split(_df2, test_size=0.4, random_state=42)
    del _df, _df2
    for i, feature_combo in enumerate(feature_combos):
        if feature_combo[0].type == _type:
            _features = [feature.name for feature in feature_combo]
            X_train = np.array(train_df[_features])
            Y_train = np.array(train_df['label'].tolist())
            X_test = np.array(test_df[_features])
            Y_test = np.array(test_df['label'].tolist())
            X2_train = np.array(train_df2[_features])
            Y2_train = np.array(train_df2['type'].tolist())
            X2_test = np.array(test_df2[_features])
            Y2_test = np.array(test_df2['type'].tolist())

            models = [
                #BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),
                #DeepConvFFNN(),
                #DeepConvStackedLSTM(),
                PassiveAggressiveClassifier(C=1.0, max_iter=100, fit_intercept=True, shuffle=False, verbose=0, loss='hinge', n_jobs=1, random_state=10, warm_start=False),
                DeepConvBiLSTM(),
                SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=100, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None,
                    random_state=10, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, n_iter_no_change=10, class_weight=None, warm_start=False, average=False),
                SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=100, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None,
                    random_state=10, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, n_iter_no_change=10, class_weight=None, warm_start=False, average=False),
                MLPClassifier(hidden_layer_sizes=(200,200), max_iter=10, alpha=0.0001, shuffle=False, verbose=0, random_state=10, warm_start=False),
            ]
            feature_set=','.join([feature.name for feature in feature_combo])
            feature_list.append(feature_set)
            #with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()/2) as executor:
            for i, model in enumerate(models):
                #final_accuracy = mp.Manager().Value("f", 0.0)
                #final_f1 = mp.Manager().Value("f", 0.0)
                algo = str(type(model).__name__)
                if algo == 'BernoulliNB':
                    algo = 'NB'
                elif algo == 'PassiveAggressiveClassifier':
                    algo = 'PA'
                elif algo == 'SGDClassifier' and model.loss == 'hinge':
                    algo = 'SVM'
                elif algo == 'SGDClassifier' and model.loss == 'log':
                    algo = 'LR'
                elif algo == 'MLPClassifier':
                    algo = 'DeepFFNN'
                #t = executor.submit(util.fitAndPredict, sampling, algo, feature_combo, model, copy.deepcopy(model),
                #                    X_train, Y_train, X_test, Y_test, X2_train, Y2_train, X2_test, Y2_test, final_accuracy, final_f1)
                final_accuracy, final_f1, final_specificity, final_sensitivity, final_auc = \
                    util.fitAndPredict(sampling, algo, [feature for feature in feature_combo], model, copy.deepcopy(model),
                                   X_train, Y_train, X_test, Y_test, X2_train, Y2_train, X2_test, Y2_test)
                if algo not in scores:
                    scores[algo] = list()
                scores[algo].append([final_accuracy, final_f1])
                if algo not in scores2:
                    scores2[algo] = list()
                scores2[algo].append([final_specificity, final_sensitivity, final_auc])
                #executor.shutdown(True)

            del X_train, Y_train
            del X_test, Y_test
            del X2_train, Y2_train
            del X2_test, Y2_test

    df = pd.DataFrame()
    for algo, score in scores.items():
        new_df = pd.DataFrame(score, columns=['accuracy', 'f1'])
        new_df = util.add_top_column(new_df, algo)
        df = pd.concat([df, new_df], axis=1)

    df.index = feature_list

    latex = df.to_latex(index=True)
    with open(str(sample_size)+'_'+sampling+'_'+str(_type)+'_'+str(min_components)+'.tex', 'w') as f:
        f.write(latex)

    df = pd.DataFrame()
    for algo, score in scores2.items():
        new_df = pd.DataFrame(score, columns=['specificity', 'sensitivity', 'auc'])
        new_df = util.add_top_column(new_df, algo)
        df = pd.concat([df, new_df], axis=1)

    df.index = feature_list

    latex = df.to_latex(index=True)
    with open(str(sample_size)+'_'+sampling+'_'+str(_type)+'_'+str(min_components)+'(2).tex', 'w') as f:
        f.write(latex)


    '''
    index = feature_list
    results = pd.DataFrame(final_accuracies, columns=column_names)
    results.index = index
    ax = results.plot(kind='bar', figsize=(50, 50), width=0.9)
    for bar in ax.patches:
        value = bar.get_height()
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + value
        ax.text(text_x, text_y, text, ha='center', color='r', size=16, rotation=90)
    ax.legend(bbox_to_anchor=(1.0, 1.0))

    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

    plt.xlabel("features", fontdict=font2)
    plt.ylabel("accuracy", fontdict=font2)
    ax.figure.savefig(str(sample_size)+'_'+sampling+'_'+str(_type)+'_'+'accuracy'+'.png')

    for i, final_accuracy in enumerate(final_accuracies):
        avg_final_accuracy = round(mean(final_accuracy), 2)
        final_accuracies[i] = avg_final_accuracy

    df = pd.DataFrame(final_accuracies)
    df.index = index
    ax = df.plot(kind='bar', figsize=(20, 20), width=0.9)
    for bar in ax.patches:
        value = bar.get_height()
        text = f'{value}'
        text_x = bar.get_x() + bar.get_width() / 2
        text_y = bar.get_y() + value
        ax.text(text_x, text_y, text, ha='center', color='r', size=16, rotation=90)
    ax.legend()


    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

    plt.xlabel("features", fontdict=font2)
    plt.ylabel("avg. accuracy", fontdict=font2)
    ax.figure.savefig(str(sample_size)+'_'+sampling+'_'+str(_type)+'_'+'avg_accuracy'+'.png')
    '''
