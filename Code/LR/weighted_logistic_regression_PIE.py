import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
from sklearn.preprocessing import label_binarize
from LR.logistic_regression import WeightedLogisticRegression
from scipy import interp


mat1 = sio.loadmat('PIE/P1.mat')
mat2 = sio.loadmat('PIE/P2.mat')
mat3 = sio.loadmat('PIE/P3.mat')
mat4 = sio.loadmat('PIE/P4.mat')
mat5 = sio.loadmat('PIE/P5.mat')
Y1 = mat1['labels']
Y2 = mat2['labels']
Y3 = mat3['labels']
Y4 = mat4['labels']
Y5 = mat5['labels']
X1 = mat1['fts']
X2 = mat2['fts']
X3 = mat3['fts']
X4 = mat4['fts']
X5 = mat5['fts']

X = [X1,X2,X3,X4,X5]
y = [Y1,Y2,Y3,Y4,Y5]
 

m = len(y)

f = open('weighted_logistic_regression_VOC.log', 'w')

wlr = WeightedLogisticRegression()

#kfold = StratifiedKFold(n_splits=5,random_state=1).split(X, y)
for fold in range(0, 5):
    Y_cross, X_cross, X_eval, Y_eval, y_bin_cross, y_bin_eval = wlr.trainEval(X, y, fold)
    label_cv = np.zeros(len(Y_cross))
    label_eval = np.zeros(len(Y_eval))
    n = np.size(X_cross,1)
    params = np.zeros((n,1))
    result =[]
    w = wlr.weight(X_cross, X_eval)

    for i in np.unique(Y_cross):
        label_cv = np.where(Y_cross == i, 1, 0)
        label_eval = np.where(Y_eval == i, 1, 0)
        params_optimal = wlr.gradient_descent(X_cross, Y_cross, params, w)
        result.append(wlr.predict(X_eval, params_optimal))

    label = wlr.evaluation(Y_eval, result)
    print(label.shape)
    class_num = np.unique(Y_eval)
    y_bin = label_binarize(Y_eval, classes=class_num)
    wlr.eval_roc_auc(y_bin, label)
    print(f"fold-{fold} finished")
f.close()