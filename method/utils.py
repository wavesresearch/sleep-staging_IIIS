#!/usr/bin/env python
# coding: utf-8
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, precision_score,recall_score,roc_auc_score,cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import random
import string

def split_dataset(X, y, val_size=0.4, test_size=0.5, _print=True):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size)
    return X_train,X_val,X_test,np_utils.to_categorical(y_train),np_utils.to_categorical(y_val), np_utils.to_categorical(y_test)

def get_callbacks(_s, _PATH_cp, callbacks=[],_logs=False):
    s_ran = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    callbacks.append(ModelCheckpoint(filepath="{0}cp_{1}_{2}.h5".format(_PATH_cp,_s, s_ran), verbose=1,save_best_only=True))
    callbacks.append(EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True))
    if _logs:
        log_dir= _PATH_cp +"logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
        callbacks.append(hp.KerasCallback(log_dir,{'num_relu_units': 512, 'dropout': 0.2}))
    return callbacks

def train_model(_PATH_cp,_s,model, X,X_val,y,y_val,e_fit,n_class=2,loss="categorical_crossentropy",_logs=False,_weights=None):
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    callbacks = get_callbacks(_s, _PATH_cp, callbacks=[])
    if _weights is not None:
        _class_weight = _weights
    else:
        _class_weight = {i:1 for i in range(n_class)}
    fittedModel = model.fit(X, y, batch_size = 16, epochs =e_fit, verbose = 2,
        validation_data=(X_val, y_val), callbacks=callbacks,
        class_weight = _class_weight, workers=20, use_multiprocessing=True)
    return model, fittedModel.history

def get_metrics(target, predicted,predicted_prob=None, cmatrix_plot=False,img_desc="Figure_x", _print=False, _title="", _average="macro", save_fig=False,cbar=True):#micro, macro
    if cmatrix_plot: confusion_matrix_heatmap(target, predicted,img_desc=img_desc, save_fig=save_fig,cbar=cbar)#, _title=_title
    _acc = round(accuracy_score(target, predicted), 3)
    _fscore = round(f1_score(target, predicted, average=_average), 3)
    _precision = round(precision_score(target, predicted, average=_average), 3)
    _recall = round(recall_score(target, predicted, average=_average), 3)
    _kappa = round(cohen_kappa_score(target, predicted),3)
    if len(set(target))==1: _auroc = 0
    elif len(set(target))>2: _auroc = round(roc_auc_score(target, predicted_prob, average="macro", multi_class="ovr"), 3)
    else: _auroc = round(roc_auc_score(target, predicted_prob[:,1],average=_average), 3)
    if _print: print( "acc: ",_acc, "fscore: ", _fscore, "precision: ", _precision, "recall: ", _recall, "auroc: ", _auroc, "kappa: ", _kappa)
    return _acc,_fscore,_precision,_recall,_auroc, _kappa

def confusion_matrix_heatmap(target, predicted, perc=False, img_desc="Figure_x",save_fig=False, cbar=True):
    plt.figure()
    data = {'y_Actual': target, 'y_Predicted': predicted}
    df = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    c_matrix_l = pd.crosstab(df['y_Predicted'], df['y_Actual'],
        rownames=['Predicted'], colnames=['Actual'])
    #print(c_matrix_l)
    if perc:
        sns.heatmap(c_matrix_l/np.sum(c_matrix_l),
            annot=True, fmt='.2%', cmap='Blues', cbar=cbar)
    else: sns.heatmap(c_matrix_l, annot=True, fmt='d')
    if save_fig:
        try:
            plt.savefig('{0}.jpeg'.format(img_desc),format='jpeg',dpi=400)
            plt.close()
        except Exception as e:
            print(e)
    return c_matrix_l

def permute_chs(X, ch_idx):
    #changes channels from one instance to another, does not change data
    print(ch_idx)
    X_permuted = X.copy()
    np.random.shuffle(X_permuted[:, ch_idx])
    return X_permuted

def local_metric(model_x, X, y):
    probs = model_x.predict(X)
    preds = probs.argmax(axis=-1)
    _acc, _fscore, _precision, _recall, _auroc, _kappa = get_metrics(y.argmax(axis=-1), preds, probs, _print=True)
    return _acc, _fscore, _precision, _recall, _auroc, _kappa

def calculate_permutation_importance(model, X, y):
    importances_acc, importances_fscore, importances_precision, importances_recall, importances_auroc, importances_kappa = [],[],[],[],[],[]
    baseline_acc, baseline_fscore, baseline_precision, baseline_recall, baseline_auroc, baseline_kappa = local_metric(model, X, y)
    for ch_id in range(X.shape[1]):
        X_permuted = permute_chs(X, ch_id)
        _acc, _fscore, _precision, _recall, _auroc, _kappa = local_metric(model, X_permuted, y)
        importances_acc.append(baseline_acc - _acc)
        importances_fscore.append(baseline_fscore - _fscore)
        importances_precision.append(baseline_precision - _precision)
        importances_recall.append(baseline_recall - _recall)
        importances_auroc.append(baseline_auroc - _auroc)
        importances_kappa.append(baseline_kappa - _kappa)
    return importances_acc, importances_fscore, importances_precision, importances_recall, importances_auroc, importances_kappa
