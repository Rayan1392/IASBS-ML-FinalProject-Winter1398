import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.optimize import newton_cg
from scipy.special import expit, exprel
from scipy.linalg import eigvalsh
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from scipy.linalg import solve_triangular
from sklearn.linear_model.logistic import ( _logistic_loss_and_grad, _logistic_loss, _logistic_grad_hess,)
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator


class BayesianLogisticRegression(LinearClassifierMixin, BaseEstimator):
    '''
    Implements Bayesian Logistic Regression with type II maximum likelihood 
    uses Gaussian method for approximation of evidence function.
    '''
    
    def __init__(self, n_iter = 50, tol = 1e-3, n_iter_solver = 15
                 , tol_solver = 1e-3, alpha = 1e-6):
        
        self.n_iter        = n_iter
        self.tol           = tol
        self.n_iter_solver     = n_iter_solver
        self.tol_solver        = tol_solver
        self.alpha             = alpha
        self._mask_val         = -1.
        

    def fit(self,X,y):
        '''
        Fits Bayesian Logistic Regression
        '''

        # preprocess data
        X,y = check_X_y( X, y , dtype = np.float64)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
               
        n_samples, n_features = X.shape
        self.coef_, self.sigma_ = [0]*n_classes,[0]*n_classes
        self.intercept_         = [0]*n_classes
        # make classifier for each class (one-vs-all) []
        for i in range(len(self.coef_)):
            pos_class = self.classes_[i]
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask] = self._mask_val
            coef_, sigma_ = self._fit(X,y_bin)
            self.coef_[i]      = coef_
            self.sigma_[i] = sigma_
            
        self.coef_  = np.asarray(self.coef_)
        return self


    def _fit(self,X,y):
        '''
        Maximizes evidence function (maximum likelihood) 
        '''
        # iterative evidence maximization
        alpha = self.alpha
        n_samples,n_features = X.shape
        w0 = np.zeros(n_features)
        
        for i in range(self.n_iter):
            
            alpha0 = alpha
            
            # mean and covariance of Laplace approximation
            w, d   = self._posterior(X, y, alpha, w0) 
            mu_sq  = np.sum(w**2)
            
            # Iterative updates
            alpha = X.shape[1] / (mu_sq + np.sum(d))
            
            # check convergence
            delta_alpha = abs(alpha - alpha0)
            if delta_alpha < self.tol or i==self.n_iter-1:
                break
            
        # find updated MAP vector
        coef_, sigma_ = self._posterior(X, y, alpha , w)
        self.alpha_ = alpha
        return coef_, sigma_
            
           
    def _posterior(self, X, Y, alpha0, w0):
        '''
        posterior function
        '''
        n_samples,n_features  = X.shape
        
        f = lambda w: _logistic_loss_and_grad(w,X[:,:-1],Y,alpha0)
        w = fmin_l_bfgs_b(f, x0 = w0, pgtol = self.tol_solver,
                            maxiter = self.n_iter_solver)[0]
        
            
        # calculate negative of Hessian at w
        xw    = np.dot(X,w)
        s     = expit(xw)
        R     = s * (1 - s)
        Hess  = np.dot(X.T*R,X)    
        Alpha = np.ones(n_features)*alpha0
        np.fill_diagonal(Hess, np.diag(Hess) + Alpha)
        e  =  eigvalsh(Hess)        
        return w,1./e


