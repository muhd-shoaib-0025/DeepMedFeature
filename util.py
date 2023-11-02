from sklearn.metrics import f1_score
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from datetime import datetime
import pytz
from sklearn.preprocessing import label_binarize
import main
import multiprocessing as mp
import threading
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_curve, auc

def add_top_column(df, top_col, inplace=False):
    if not inplace:
        df = df.copy()
    df.columns = pd.MultiIndex.from_product([[top_col], df.columns])
    return df

def undersample(df):
    X = df.drop('label', axis=1)
    Y = df['label']
    under_sampler  = RandomUnderSampler(random_state=42)
    X, Y = under_sampler.fit_resample(X, Y)
    X['label'] = Y

    X2 = df.query('label == 1').drop('type', axis=1)
    Y2 = df.query('label == 1')['type']
    under_sampler = RandomUnderSampler(random_state=42)
    X2, Y2 = under_sampler.fit_resample(X2, Y2)
    X2['type'] = Y2

    return pd.concat([X, X2])

def oversample(df):
    X = df.drop('label', axis=1)
    Y = df['label']
    over_sampler  = RandomOverSampler(random_state=42)
    X, Y = over_sampler.fit_resample(X, Y)
    X['label'] = Y

    X2 = df.query('label == 1').drop('type', axis=1)
    Y2 = df.query('label == 1')['type']
    over_sampler  = RandomOverSampler(random_state=42)
    X2, Y2 = over_sampler.fit_resample(X2, Y2)
    X2['type'] = Y2

    return pd.concat([X, X2])

def getrows(X_train, Y_train, chunkrows):
    return X_train[chunkrows], Y_train[chunkrows]

def iter_minibatches(X_train, Y_train, chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    numtrainingpoints = X_train.shape[0]
    while chunkstartmarker + chunksize < numtrainingpoints:
        chunkrows = range(chunkstartmarker, chunkstartmarker + chunksize)
        X_chunk, y_chunk = getrows(X_train, Y_train, chunkrows)
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize

def runFold(sampling, algo, feature_combo, fold, stage, model, X_train, Y_train, X_val, Y_val, batch_size):
    timezone = pytz.timezone("Asia/Karachi")
    local_time = datetime.now(timezone)
    current_time = local_time.strftime("%I:%M:%S%p")

    x_dims = [feature_combo[i].feature_size for i in range(len(feature_combo))]
    x_dim_max = max(x_dims)
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    for i, x_dim in enumerate(x_dims):
        size_diff = x_dim_max-x_dim
        df_train = pd.concat([df_train, pd.DataFrame(X_train[:, i].tolist())], axis=1)
        df_val = pd.concat([df_val, pd.DataFrame(X_val[:, i].tolist())], axis=1)
        if algo in ['ConvFFNN', 'ConvStackedLSTM', 'ConvBiLSTM']:
            padding_train = pd.DataFrame(np.array([0 for i in range(size_diff * len(X_train))]).reshape(len(X_train), size_diff))
            padding_val = pd.DataFrame(np.array([0 for i in range(size_diff * len(X_val))]).reshape(len(X_val), size_diff))
            df_train = pd.concat([df_train, padding_train], axis=1)
            df_val = pd.concat([df_val, padding_val], axis=1)

    y_dim = len(feature_combo)
    X_train = np.asarray(df_train)
    X_val = np.asarray(df_val)
    if algo in ['ConvFFNN', 'ConvStackedLSTM', 'ConvBiLSTM']:
        X_train = X_train.reshape(len(X_train), y_dim, x_dim_max)
        X_val = X_val.reshape(len(X_val), y_dim, x_dim_max)
        model.fit(X_train, Y_train, batch_size)
    else:
        batch_size = min(int(X_train.shape[0] / 2), batch_size)
        batcherator = iter_minibatches(X_train, Y_train, chunksize=batch_size)
        for i in range(100):
            for X_chunk, y_chunk in batcherator:
                model.partial_fit(X_chunk, y_chunk, classes=np.unique(Y_train))

    Y_pred = model.predict(X_val)
    train_accuracy = accuracy_score(Y_val[0:len(Y_pred)], Y_pred)

    print(
        'sampling: ', sampling,
        '; algo: ', algo,
        '; features: ', [feature.name for feature in feature_combo],
        '; shape: ', X_train.shape,
        '; fold (stage-%s): ' %stage , fold,
        '; time: ', current_time,
        '; cpu: ', mp.current_process().name,
        '; thread: ', threading.get_ident()
    )

    return train_accuracy

def fitAndPredict(
        sampling, algo, feature_combo, model, model2, X_train, Y_train, X_test, Y_test, X2_train, Y2_train, X2_test, Y2_test, batch_size):
    kf = KFold(n_splits=main.num_folds)
    fold=1
    best_accuracy = 0.0

    for train_index, test_index in kf.split(X_train):
        X_t, X_v = X_train[train_index], X_train[test_index]
        Y_t, Y_v = Y_train[train_index], Y_train[test_index]
        train_accuracy = runFold(sampling, algo, feature_combo, fold, 1, model, X_t, Y_t, X_v, Y_v, batch_size)
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_model = model
            break
        fold += 1

    x_dims = [feature_combo[i].feature_size for i in range(len(feature_combo))]
    x_dim_max = max(x_dims)
    df_test = pd.DataFrame()
    df_test2 = pd.DataFrame()
    for i, x_dim in enumerate(x_dims):
        size_diff = x_dim_max - x_dim
        df_test = pd.concat([df_test, pd.DataFrame(X_test[:, i].tolist())], axis=1)
        df_test2 = pd.concat([df_test2, pd.DataFrame(X2_test[:, i].tolist())], axis=1)
        if algo in ['ConvFFNN', 'ConvStackedLSTM', 'ConvBiLSTM']:
            padding_test = pd.DataFrame(
                np.array([0 for i in range(size_diff * len(X_test))]).reshape(len(X_test), size_diff))
            padding_test2 = pd.DataFrame(
                np.array([0 for i in range(size_diff * len(X2_test))]).reshape(len(X2_test), size_diff))
            df_test = pd.concat([df_test, padding_test], axis=1)
            df_test2 = pd.concat([df_test2, padding_test2], axis=1)

    y_dim = len(feature_combo)
    X_test = np.asarray(df_test)
    X2_test = np.asarray(df_test2)
    if algo in ['ConvFFNN', 'ConvStackedLSTM', 'ConvBiLSTM']:
        X_test = X_test.reshape(len(X_test), y_dim, x_dim_max)
        X2_test = X2_test.reshape(len(X2_test), y_dim, x_dim_max)

    fold=1
    best_accuracy2 = 0.0
    for train_index, test_index in kf.split(X2_train):
        X_t, X_v = X2_train[train_index], X2_train[test_index]
        Y_t, Y_v = Y2_train[train_index], Y2_train[test_index]
        train_accuracy2 = runFold(sampling, algo, feature_combo, fold, 2, model2, X_t, Y_t, X_v, Y_v, batch_size)
        fold += 1
        if train_accuracy2 > best_accuracy2:
            best_accuracy2 = train_accuracy2
            best_model2 = model2
            break

    Y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test[0:len(Y_pred)], Y_pred)
    f1_1 = f1_score(Y_test[0:len(Y_pred)], Y_pred, average='weighted')

    Y2_pred = best_model2.predict(X2_test)
    test_accuracy2 = accuracy_score(Y2_test[0:len(Y2_pred)], Y2_pred)
    f1_2 = f1_score(Y2_test[0:len(Y2_pred)], Y2_pred, average='weighted')

    final_accuracy = float("{:.2f}".format(test_accuracy * test_accuracy2 * 100))
    final_f1 = float("{:.2f}".format(f1_1 * f1_2 * 100))

    tn, fp, fn, tp = confusion_matrix(Y_test[0:len(Y_pred)], Y_pred).ravel()
    specificity_1 = tn / (tn + fp)
    sensitivity_1 = tp / (tp + fn)

    le = preprocessing.LabelEncoder()
    Y2_test = le.fit_transform(Y2_test)
    Y2_pred = le.fit_transform(Y2_pred)
    c_m = confusion_matrix(Y2_test[0:len(Y2_pred)], Y2_pred)
    fn = c_m.sum(axis=1) - np.diag(c_m)
    fp = c_m.sum(axis=0) - np.diag(c_m)
    tn = c_m.sum() - (fp + fn + np.diag(c_m))
    tp = np.diag(c_m)
    specificity_2 = tn / (tn + fp)
    specificity_2 = sum(specificity_2) / len(specificity_2)
    sensitivity_2 = tp / (tp + fn)
    sensitivity_2 = sum(sensitivity_2) / len(sensitivity_2)

    final_specificity = float("{:.2f}".format(specificity_1 * specificity_2 * 100))
    final_sensitivity = float("{:.2f}".format(sensitivity_1 * sensitivity_2 * 100))

    Y2_test = label_binarize(Y2_test[0:len(Y2_pred)], classes=range(4))
    Y2_pred = label_binarize(Y2_pred, classes=range(4))
    fpr_micro, tpr_micro, _ = roc_curve(Y2_test.ravel(), Y2_pred.ravel())
    final_roc_auc = auc(fpr_micro, tpr_micro)

    return final_accuracy, final_f1, final_specificity, final_sensitivity, final_roc_auc

