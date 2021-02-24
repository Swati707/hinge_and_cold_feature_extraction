# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:29:46 2020

@author: swati
"""
import argparse
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import *
import numpy as np

if __name__ == '__name__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--hinge_features", type=str, default=r"E:\swati-plp\gender_identification\existing_papers\gender_identification_gattal_et_al\data\our_dataset\hinge_features.npy")
    parser.add_argument("--cold_features", type=str, default=r"E:\swati-plp\gender_identification\existing_papers\gender_identification_gattal_et_al\data\our_dataset\cold_features.npy")
    parser.add_argument("--gt_label", type=str, default=r"E:\swati-plp\gender_identification\existing_papers\gender_identification_gattal_et_al\data\our_dataset\labels.npz")
    opt = parser.parse_args()
    
    x_hinge = np.load(opt.hinge_features)
    x_cold = np.load(opt.cold_features)
    y = np.load(opt.gt_label)['label']
    label_names = np.load(opt.gt_label)['label_name']
    
    cs = MinMaxScaler()
    x = cs.fit_transform(x_hinge) 
    x = np.nan_to_num(x)
    clf = SVC(kernel='rbf', verbose=True, C=10)
    # scores = cross_val_score(clf, x, y, cv=10)
    # print(scores)
    y_pred_hinge = cross_val_predict(clf, x, y, cv=10)
    
    cs = MinMaxScaler()
    x_ = cs.fit_transform(x_cold)
    x_ = np.nan_to_num(x_)
    clf = SVC(kernel='rbf', verbose=True, C=10)
    # scores = cross_val_score(clf, x_, y, cv=10)
    # print(scores)
    y_pred_cold = cross_val_predict(clf, x_, y, cv=10)
    
    y_pred = np.maximum(y_pred_hinge, y_pred_cold)
    
    print('Confusion Matrix')
    cf = confusion_matrix(y, y_pred)
    cf_sum = cf.sum(axis = 1)[:, np.newaxis]
    cf = np.round(cf / cf_sum * 100, 2)
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print(repr(cf))
    
    cr = 0.0
    for i in range(0, cf.shape[0]):
        cr += cf[i][i]
        
    cr /= cf.shape[0]
    print(f'classification rate = {np.round(cr, 2)}')
    
    print(classification_report(y, y_pred))
