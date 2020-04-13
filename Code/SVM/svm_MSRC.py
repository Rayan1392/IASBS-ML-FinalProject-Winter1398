import numpy as np
import scipy.io as sio
from sklearn.preprocessing import label_binarize
from SVM.svm import SVM
    

mat = sio.loadmat('MSRC/MSRC.mat')

y = mat['labels']
X = mat['fts']

class_num = np.unique(y)
y_bin = label_binarize(y, classes=class_num)

svm = SVM()
svm.svc(X, y, y_bin)
    
