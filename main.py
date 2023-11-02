def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import argparse
import copy
import math
from sklearn.neural_network import MLPClassifier
from collections import OrderedDict
import util
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn import preprocessing
import feature
from itertools import combinations
from sklearn.feature_extraction.text import TfidfTransformer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.layers import BatchNormalization
import pandas as pd
from keras.layers import Conv1D, Conv2D
from keras.models import Model
from keras.utils.data_utils import *
from autoencoder import *
from keras.layers import Dense, Input
from keras.layers import Flatten
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
global sample_size
global sampling

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
k=10000
num_folds=10

class DL:

    def __init__(self, n_iter=10):
        self.n_iter = n_iter
        self.batch_size = None
        self.model = None

    def fit(self, train_x, train_y, batch_size):

        num_batches = int(train_x.shape[0]/batch_size)
        train_x = train_x[0:num_batches * batch_size, :].reshape(num_batches, batch_size, train_x.shape[1], train_x.shape[2])
        train_y = train_y[0:num_batches * batch_size].reshape(num_batches, batch_size)
        input_layer, output_layer = self.get_input_output_layer(train_x, train_y)
        model = Model(inputs=input_layer, outputs=output_layer, )
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=True)
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

class ConvFFNN(DL):
    def get_input_output_layer(self, train_x, train_y):

        input_layer = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)
        bn2 = tf.reshape(bn2, [bn2.shape[1], bn2.shape[2], bn2.shape[3]])
        do = Dropout(0.1)(bn2)
        flat = Flatten()(do)
        output_layer = Dense(len(np.unique(train_y)), activation='softmax')(flat)
        return input_layer, output_layer

class ConvStackedLSTM(DL):
    def get_input_output_layer(self, train_x, train_y):

        input_layer = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)
        bn2 = tf.reshape(bn2, [bn2.shape[1], bn2.shape[2], bn2.shape[3]])
        lstm1 = LSTM(num_hidden, return_sequences=True)(bn2)
        do1 = Dropout(0.1)(lstm1)
        lstm2 = LSTM(num_hidden)(do1)
        do2 = Dropout(0.1)(lstm2)
        flat = Flatten()(do2)
        output_layer = Dense(len(np.unique(train_y)), activation='softmax')(flat)
        return input_layer, output_layer

class ConvBiLSTM(DL):
    def get_input_output_layer(self, train_x, train_y):
        ConvStackedLSTM(),
        input_layer = Input(shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3]))
        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv2)
        bn2 = tf.reshape(bn2, [bn2.shape[1], bn2.shape[2], bn2.shape[3]])
        lstm1 = Bidirectional(LSTM(num_hidden, return_sequences=True))(bn2)
        do = Dropout(0.1)(lstm1)
        flat = Flatten()(do)
        output_layer = Dense(len(np.unique(train_y)), activation='softmax')(flat)
        return input_layer, output_layer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang", type=str, required=True, help="Specify the language.")
    parser.add_argument("-sample_size", type=float, required=True, help="Specify sample size as a float.")
    parser.add_argument("-num_hidden", type=int, required=True, help="Specify the hidden layer size.")
    parser.add_argument("-min_comp", type=int, required=True, help="Specify the minimum components as an integer.")
    parser.add_argument("-batch_size", type=int, required=True, help="Specify the batch_size.")
    parser.add_argument("-num_threads", type=int, required=True, help="Specify the number of threads as an integer.")
    parser.add_argument("-sampling", type=str, required=True, help="Specify the sampling method.")

    args = parser.parse_args()
    return args.lang, args.sample_size, args.num_hidden, args.min_comp, args.batch_size, args.num_threads, args.sampling

import multiprocessing as mp

if __name__ == '__main__':

    global sampling
    lang, sample_size, num_hidden, min_comp, batch_size, num_threads, sampling = parse_arguments()

    smiles = feature.Smiles("SMILES", k, 1, num_threads)
    biobert_en = feature.BioBert_en("Bio-Bert_en", k, 1, num_threads)
    rdf2vec = feature.RDF2Vec("RDF2Vec", k, 1, num_threads)
    frequent_words = feature.FrequentWords("Frequent Words", k, 0, num_threads)
    frequent_phrases = feature.FrequentPhrases("Frequent Phrases", k, 0, num_threads)
    frequent_syntactic_trios = feature.SyntacticGrammarTrios("Frequent Syntactic Trios", k, 0, num_threads)
    biobert_ru = feature.BioBert_ru("Bio-Bert_ru", k, 1, num_threads)
    fasttext_ru = feature.Fasttext_ru("Fasttext_ru", k, 1, num_threads)

    try:
        os.mkdir('results')
    except:
        pass

    if lang == 'ru':
        features = [
            fasttext_ru,
            biobert_ru,
            frequent_words,
            frequent_phrases,
        ]
    else:
        features = [
            smiles,
            rdf2vec,
            biobert_en,
            frequent_words,
            frequent_phrases,
            frequent_syntactic_trios,
        ]
    df = pd.read_excel('dataset/DDIdataset-' + lang + '.xlsx', engine='openpyxl', sheet_name='Sheet')
    df = df.sample(frac=sample_size, replace=True, random_state=42).reset_index(drop=True)
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    feature_combos = list()
    for i in range(1, len(features) + 1):
        all_feature_combos = list(combinations(features, i))
        for feature_combo in all_feature_combos:
            if len(set([feature.type for feature in feature_combo])) == 1:
                feature_combos.append(feature_combo)
    feature_combos.sort(key=len)
    feature_combos.reverse()

    for feature in features:
        feature_matrix = feature.create_df(df)
        feature_matrix = pd.DataFrame.from_dict(feature_matrix, orient='index')
        original_index = feature_matrix.index
        if feature.type == 0:
            feature_matrix = TfidfTransformer().fit_transform(np.asarray(feature_matrix)).toarray()
        num_batches = math.ceil(10 * sample_size)
        num_components = min(feature_matrix.shape[1], int(feature_matrix.shape[0] / num_batches), min_comp)
        incremental_pca = IncrementalPCA(n_components=num_components)
        for batch in np.array_split(feature_matrix, num_batches):
            incremental_pca.fit(batch)
        feature_matrix = incremental_pca.transform(feature_matrix)
        transformed_df = pd.DataFrame()
        transformed_df[feature.name] = pd.DataFrame(data=feature_matrix, index=original_index).apply(lambda row: row.tolist(), axis=1)
        df = pd.concat([df, transformed_df], axis=1).dropna(subset=[feature.name])
        feature_size = len(feature_matrix[0])
        feature.set_feature_size(feature_size)

    if sampling == 'under_sampling':
        _df = util.undersample(df)
    elif sampling == 'over_sampling':
        _df = util.oversample(df)
    else:
        _df = df

    print('dataset stats:')
    print('number of instances', len(_df))
    print('number of positive DDIs', len(_df.query("label == 1")))
    print('number of negative DDIs', len(_df.query("label == 0")))
    if lang == 'ru':
        print('number of Advise DDIs', len(_df.query('type == "советовать"')))
        print('number of Effect DDIs', len(_df.query('type == "эффект"')))
        print('number of Mechanism DDIs', len(_df.query('type == "механизм"')))
        print('number of Int DDIs', len(_df.query('type == "интервал"')))
    else:
        print('number of Advise DDIs', len(_df.query('type == "advise"')))
        print('number of Effect DDIs', len(_df.query('type == "effect"')))
        print('number of Mechanism DDIs', len(_df.query('type == "mechanism"')))
        print('number of Int DDIs', len(_df.query('type == "int"')))

    _df2 = _df.query("label == 1")
    _df2['type'] = le.fit_transform(_df2['type'])

    train_df, test_df = train_test_split(_df, test_size=0.4, random_state=42)
    train_df2, test_df2 = train_test_split(_df2, test_size=0.4, random_state=42)
    del _df, _df2
    batch_size = min(int(train_df2.shape[0] / 2), batch_size)
    if train_df2.shape[0] < batch_size:
        batch_size = int(train_df2.shape[0] / 2)
    result_df = pd.DataFrame()
    feature_list = list()
    scores = OrderedDict()
    for _type in [1, 0]:
        filename = f'{sampling}{sample_size}_hidden{num_hidden}_batch{batch_size}_comp{min_comp}'
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
                    MLPClassifier(hidden_layer_sizes=(num_hidden, num_hidden), max_iter=10, alpha=1.0, shuffle=False, verbose=0, random_state=10, warm_start=False,),
                    ConvFFNN(),
                    ConvBiLSTM(),
                    ConvStackedLSTM(),
               ]
                feature_set=','.join([feature.name for feature in feature_combo])
                feature_list.append(feature_set)

                for i, model in enumerate(models):
                    algo = str(type(model).__name__)
                    if algo == 'MLPClassifier':
                        algo = 'DeepFFNN'
                    final_accuracy, final_f1, final_specificity, final_sensitivity, final_roc = util.fitAndPredict(
                        sampling, algo, [feature for feature in feature_combo], model, copy.deepcopy(model), X_train, Y_train, X_test, Y_test, X2_train, Y2_train, X2_test, Y2_test, batch_size)
                    if algo not in scores:
                        scores[algo] = list()
                    scores[algo].append([final_accuracy, final_f1, final_specificity, final_sensitivity, final_roc])

                del X_train, Y_train
                del X_test, Y_test
                del X2_train, Y2_train
                del X2_test, Y2_test

        import matplotlib.pyplot as plt

        feature_index = max(range(len(feature_list)), key=lambda i: len(feature_list[i].split(',')))

        specificities = list()
        sensitivities = list()
        roc_aucs = list()
        accuracies = list()
        for algo, score in scores.items():
            accuracies.append([s[0] for s in score])
            specificities.append([s[2] for s in score])
            sensitivities.append([s[3] for s in score])
            roc_aucs.append([s[4] for s in score])

        if _type == 1:
            roc_aucs = list(zip(*roc_aucs))

            plt.figure()
            for i, roc_auc in enumerate(roc_aucs):
                plt.plot([algo for algo, score in scores.items()], roc_auc, label=feature_list[i])

            plt.title('ROC curve')
            plt.legend(loc='lower right')
            plt.xticks(rotation=45)
            plt.ylim([0, 1.1])
            plt.tight_layout()
            plt.savefig(filename + '_' + 'ROC' + '_' + lang + '.png')

            sensitivities = list(zip(*sensitivities))

            plt.figure()
            for i, sensitivity in enumerate(sensitivities):
                plt.plot([algo for algo, score in scores.items()], sensitivity, label=feature_list[i])

            plt.title('Sensitivity')
            plt.legend(loc='lower right')
            plt.xticks(rotation=45)
            plt.ylim([0, 110])
            plt.tight_layout()
            plt.savefig(filename + '_' + 'Sensitivity' + '_' + lang + '.png')

            specificities = list(zip(*specificities))

            plt.figure()
            for i, specificity in enumerate(specificities):
                plt.plot([algo for algo, score in scores.items()], specificity, label=feature_list[i])

            plt.title('Specificity')
            plt.legend(loc='lower right')
            plt.xticks(rotation=45)
            plt.ylim([0, 110])
            plt.tight_layout()
            plt.savefig(filename + '_' + 'Specificity' + '_' + lang + '.png')

            plt.figure()
            plt.boxplot(accuracies, vert=True, patch_artist=True)
            plt.xticks(range(1, len(models) + 1), [algo for algo, score in scores.items()], rotation=45)
            plt.title('Accuracy Box Plot')
            plt.ylim([0, 110])
            plt.tight_layout()
            plt.savefig(filename + '_' + 'boxplot' + '_' + lang + '.png')

    for algo, score in scores.items():
        new_df = util.add_top_column(pd.DataFrame([s[1] for s in score], columns=['']), algo)
        result_df = pd.concat([result_df, new_df], axis=1)

    result_df.index = feature_list

    latex = result_df.to_latex(index=True)
    with open(filename + '_' + lang + '.tex', 'w') as f:
        f.write(latex)