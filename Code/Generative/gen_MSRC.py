import numpy as np
import scipy.io as sio
from Generative.generative import Generative



mat = sio.loadmat('MSRC/MSRC.mat')

# training dataset
X = mat['fts']
# target dataset
Y = mat['labels']

n = np.size(X,1)

gen = Generative(n)
gen.pred_eval(X, Y)


