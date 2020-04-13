import logging
import numpy as np
import scipy.io as sio
from BayesLog.bayesian_logistic_regression import BayesianLogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split                           


logging.basicConfig(filename='bayes_logistic.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.debug("b_logistic_regression_MSRC");   

mat = sio.loadmat('MSRC/MSRC.mat')

X = mat['fts']
y = mat['labels']

blr = BayesianLogisticRegression()
blr.fit(X, y)

print("\n === MSRC: Bayesian Logistic Regression ===")
print(classification_report(y,blr.predict(X)))

