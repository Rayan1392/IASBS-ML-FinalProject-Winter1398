import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
from sklearn.preprocessing import label_binarize
from LR.logistic_regression import WeightedLogisticRegression
from scipy import interp

mat = sio.loadmat('MSRC/MSRC.mat')

y = mat['labels']
X = mat['fts']

class_num = np.unique(y)

m = len(y)

n = np.size(X,1)
params = np.zeros((n,1))

f = open('weighted_logistic_regression_VOC.log', 'w')

wlr = WeightedLogisticRegression()

kfold = StratifiedKFold(n_splits=5,random_state=1).split(X, y)
for fold, (train_index, eval_index) in enumerate(kfold):
    X_cross, X_eval = X[train_index], X[eval_index]
    Y_cross, Y_eval = y[train_index], y[eval_index]
    label_cv = np.zeros(len(Y_cross))
    label_eval = np.zeros(len(Y_eval))
    result =[]
    w = wlr.weight(X_cross, X_eval)

    for i in np.unique(Y_cross):
        label_cv = np.where(Y_cross == i, 1, 0)
        label_eval = np.where(Y_eval == i, 1, 0)
        params_optimal = wlr.gradient_descent(X_cross, Y_cross, params, w)
        result.append(wlr.predict(X_eval, params_optimal))

    label = wlr.evaluation(Y_eval, result)
    print(label.shape)
    y_bin = label_binarize(Y_eval, classes=class_num)
    wlr.eval_roc_auc(y_bin, label)
    #f.write('\n' + "--"*20)
    #f.write('\n'  + "Optimal Parameters are:" + '\n'+ str(result) )
    print(f"fold-{fold} finished")
f.close()

