import numpy as np
import scipy.io as sio
from Generative.generative import NiveBayesGenerative as NBGenerative



mat = sio.loadmat('VOC/VOC.mat')

# training dataset
X = mat['fts']
# target dataset
Y = mat['labels']

n = np.size(X,1)

gen = NBGenerative(n)
gen.pred_eval(X, Y)


