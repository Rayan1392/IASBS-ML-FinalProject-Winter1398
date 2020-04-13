import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, auc, roc_curve
from itertools import cycle
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp


class SVM(object):
    """implementation of SVM algorithm"""


    def __init__(self, *args, **kwargs): 
        return super().__init__(*args, **kwargs)


    def evaluation(self, Y_eval, prediction):
        '''
        evaluation function
        '''
        print(classification_report(Y_eval, prediction))


    def eval_roc_auc(self, y_test, y_score, y_bin):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_bin.shape[1]
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


    def svc(self, X, y, y_bin):
        '''
        svc function
        '''
        logging.basicConfig(filename='svm.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        logger.debug(f"svm"); 

        kfold = StratifiedKFold(n_splits=5,random_state=1).split(X, y)
        for fold, (train_index, eval_index) in enumerate(kfold):
            X_cross, X_eval = X[train_index], X[eval_index]
            Y_cross, Y_eval = y[train_index], y[eval_index]
            y_bin_cross, y_bin_eval = y_bin[train_index], y_bin[eval_index]
            clf = OneVsRestClassifier(SVC(gamma='auto', probability=True)).fit(X_cross, Y_cross)
            y_pred = np.zeros((np.size(X_eval,0)))
            y_true = np.zeros((np.size(X_eval, 0)))
            prediction = clf.predict(X_eval)
            y_score = clf.fit(X_cross, y_bin_cross).decision_function(X_eval)
            logger.info('\n' + "--"*20)
            self.evaluation(Y_eval, prediction)
            self.eval_roc_auc(y_bin_eval, y_score, y_bin)

class PIESVM(SVM):
    
    def evaluation(self, Y_eval, prediction):
        print(classification_report(Y_eval, prediction))
        print("---"*20)

    def trainEval(self, X, y, fold):
        X_eval = X[fold]
        Y_eval = y[fold]
        num_eval = np.unique(Y_eval)
        y_bin_eval = label_binarize(Y_eval, classes=num_eval)
        x_temp = []
        y_temp = []
        #y_bin_temp
        v = 0
        while v < 5 :
            if v != fold :
                x_temp.append(X[v])
                y_temp.append(y[v])
                #y_bin_temp.append(y_bin[v])
            v += 1
    
        X_cross = np.concatenate([x_temp[0],x_temp[1],x_temp[2],x_temp[3]])
        Y_cross = np.concatenate([y_temp[0],y_temp[1],y_temp[2],y_temp[3]])
        class_num = np.unique(Y_cross)
        y_bin_cross = label_binarize(Y_cross, classes=class_num)
        return Y_cross, X_cross, X_eval, Y_eval, y_bin_cross, y_bin_eval


    def svc(self, X, y):

        logging.basicConfig(filename='svm.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        logger.debug(f"svm"); 
        for fold in range(0, 5):
            Y_cross, X_cross, X_eval, Y_eval, y_bin_cross, y_bin_eval = self.trainEval(X, y, fold)
            clf = OneVsRestClassifier(SVC(gamma='auto', probability=True)).fit(X_cross, Y_cross)
            prediction = clf.predict(X_eval)
            y_score = clf.fit(X_cross, y_bin_cross).decision_function(X_eval)
            logger.info('\n' + "--"*20)
            logger.info('\n' + str(prediction))
            classification_report(Y_eval, prediction)
            self.eval_roc_auc(y_bin_eval, y_score)

    
  
