import logging
import numpy as np
import scipy.io as sio
from BayesLog.bayesian_logistic_regression import BayesianLogisticRegression
from sklearn.metrics import classification_report


logging.basicConfig(filename='bayes_logistic.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.debug("b_logistic_regression_PIE"); 

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

X = np.concatenate([X1,X2,X3,X4,X5])
y = np.concatenate([Y1,Y2,Y3,Y4,Y5])


blr = BayesianLogisticRegression()
blr.fit(X, y)

print("\n === PIE: Bayesian Logistic Regression ===")
print(classification_report(y, blr.predict(X)))
