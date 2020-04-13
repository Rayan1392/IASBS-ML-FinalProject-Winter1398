import logging
import numpy as np
import scipy.io as sio
from scipy import stats
from BayesLog.bayesian_logistic_regression import BayesianLogisticRegression
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



logging.basicConfig(filename='bayes_logistic.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.debug("b_logistic_regression_VOC");                       


mat = sio.loadmat('VOC/VOC.mat')

X = mat['fts']
y = mat['labels']

blr = BayesianLogisticRegression()
blr.fit(X, y)

print("\n === VOC: Bayesian Logistic Regression ===")
print(classification_report(y,blr.predict(X)))