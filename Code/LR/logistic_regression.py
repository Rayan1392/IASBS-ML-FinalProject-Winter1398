#from numba import jit, cuda 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio
from sklearn.metrics import roc_auc_score, auc, roc_curve
from itertools import cycle
from sklearn.preprocessing import label_binarize
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy import interp
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator


class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    """
    Superclass for two implementations of Logistic Regression: simple and weighted
    """
    def __init__ (self, iterations = 1500, learning_rate = 200):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        '''
        sigmoid function
        '''
        return 1 / (1 + np.exp(-x))


    def gradient_descent(self, X, y, params):
        raise NotImplementedError

    #@jit(target ="cuda")  
    def predict(self, X_eval, params):
        """
        return sigmid function
        """
        return self.sigmoid(X_eval @ params)

    #@jit(nopython=True, target='cuda', boundscheck=False)   
    def evaluation(self, Y_eval, result):
        """
        evaluation function
        """
        result = np.array(result)
        result = result.T
        result = result[0]
    
        label = np.argmax(result, 1)+1
        print(classification_report(Y_eval, label))
        return result

    #@jit(nopython=True, target='cuda', boundscheck=False)   
    def eval_roc_auc(self, y_test, y_score):
        """
        eval ROC_AUC
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # n_classes = y_bin.shape[1]
        n_classes = y_test.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
 
    def trainEval(self, X, y, fold):
        """
        train evaluation
        """
        X_eval = X[fold]
        Y_eval = y[fold]
        x_temp = []
        y_temp = []
        v = 0
        while v < 5 :
            if v != fold :
                x_temp.append(X[v])
                y_temp.append(y[v])
            v += 1
        X_cross = np.concatenate([x_temp[0],x_temp[1],x_temp[2],x_temp[3]])
        Y_cross = np.concatenate([y_temp[0],y_temp[1],y_temp[2],y_temp[3]])
        class_num = np.unique(Y_cross)
        y_bin_cross = label_binarize(Y_cross, classes=class_num)
        y_bin_eval = label_binarize(Y_eval, classes=class_num)
        return Y_cross, X_cross, X_eval, Y_eval, y_bin_cross, y_bin_eval


class SimpleLogisticRegression(LogisticRegression):
    """Linear Logistic Regression"""
    def __init__(self, iterations = 1500, learning_rate = 200):
        self.iterations = iterations
        self.learning_rate = learning_rate

    #@jit(nopython=True, target='cuda', boundscheck=False)  
    def gradient_descent(self, X, y, params):
        m = len(y)

        for i in range(self.iterations):
            params = params - (self.learning_rate/m) * (X.T @ (self.sigmoid(X @ params) - y))
        return  params


class WeightedLogisticRegression(LogisticRegression):
    """
    Weighted Logistic Regression
    """
    def __init__(self, iterations = 1500, learning_rate = 200):
        self.iterations = iterations
        self.learning_rate = learning_rate


    def gradient_descent(self, X, y, params, w):
        m = len(y)

        vec = []
        '''for j in range(len(w)):
            vec.append(w[j][j])
        vec = np.array(vec)'''

        for i in range(self.iterations):
            params = params - (self.learning_rate/m) *(w * (X.T @ (self.sigmoid(X @ params) - y)))
        
        return params


    def weight(self, train_x, test_x):
        w = []
        n = np.size(test_x,1)
        for i in range(len(test_x)):
            vector =[]
            exp = 0
            for j in range(len(train_x)):
            
                exp += -1* np.square( np.linalg.norm( test_x[i] - train_x[j]))# / (2*0.8*0.8) 
        
            w.append(np.exp(exp ))
        w = np.array(w)
    
        return w

