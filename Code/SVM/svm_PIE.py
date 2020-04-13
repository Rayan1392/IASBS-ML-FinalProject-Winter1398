import numpy as np
import scipy.io as sio
from sklearn.preprocessing import label_binarize
from SVM.svm import PIESVM as SVM
    

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

svm = SVM()
svm.svc(X, y)
    
