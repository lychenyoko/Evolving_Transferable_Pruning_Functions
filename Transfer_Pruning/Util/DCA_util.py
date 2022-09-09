#---------------------------
# discriminant_analysis.py
# Author: Thee Chanyaswad
#
# Version 1.1: 08/16/16
#   - Re-implement DCA to solve the eig-decomposition faster using eigh function.
#   - Old DCA renamed as DCA_v0.
# Version 1.2: 10/14/16
#   - Correct the regularization term in KDCA.
#   - Old (incorrect) implementation renamed as KDCA_v0.
# Version 2.0: 11/16/16
#   - Correct the _get_Kmatrices and transform in KDCA
#   - Old implementation renamed as KDCA_v1
# Version 2.1: 11/17/16
#   - Re-implement the centering method to fix asymmetric gram matrix problem (and negative eigval of gram matrix problem)
#   - Re-implement the calculation of Kw for faster computation
# Version 2.2: 11/18/16
#   - Use the symmetric mapping to cvs to solve for kdca instead in order to solve negative eigval of Kbar2 and Kbar problem.
#   - Move argument input to the init function
# Version 3.0: 2/2/17
#   - Add KPCA
# Version 3.1: 3/30/17
#   - Fix svd did not converge with Laplacian kernel by using 'gesvd' instead.
#
#---------------------------

import numpy as np
import scipy
from sklearn.metrics import pairwise
from sklearn import preprocessing


class DCA:
    def __init__(self, rho=None, rho_p=None, n_components=None):
        self.n_components = n_components
        self.rho = rho
        self.rho_p = rho_p

    def fit(self, X, y):
        (self._Sw, self._Sb) = self._get_Smatrices(X,y)

        if self.rho == None:
            s0 = np.linalg.eigvalsh(self._Sw)
            self.rho = 0.02*np.max(s0)
        if self.rho_p == None:
            self.rho_p = 0.1*self.rho
        
        pSw = self._Sw + self.rho*np.eye(self._Sw.shape[0])
        pSbar = self._Sb + self._Sw + (self.rho_p+self.rho)*np.eye(self._Sw.shape[0])

        (s1,vr) = scipy.linalg.eigh(pSbar,pSw,overwrite_a=True,overwrite_b=True)
        s1 = s1[::-1] #re-order from large to small
        Wdca = vr.T[::-1]
        self.eigVal = s1
        self.allComponents = Wdca
        if self.n_components:
            self.components = Wdca[0:self.n_components]
        else:
            self.components = Wdca


    def transform(self, X, dim=None):
        if dim == None:
            X_trans = np.inner(self.components,X)
        else:
            X_trans = np.inner(self.allComponents[0:dim],X)
        return X_trans.T

    def inverse_transform(self, Xreduced, projMatrix=None, dim=None):
        if projMatrix is None:
            if dim is None:
                W = self.components
            else:
                W = self.allComponents[0:dim]
        else:
            W = projMatrix
        #W = PxM where P<M
        foo = np.inner(W,W)
        bar = np.linalg.solve(foo.T,W)
        Xhat = np.inner(Xreduced,bar.T)
        return Xhat

    def _get_Smatrices(self, X,y):
        '''
        Usage:
            X: (np.array) of data matrix in the shape of [N, d]
            y: (np.array) of labels in the shape of [N] or [N,1]
        '''
        Sb = np.zeros((X.shape[1],X.shape[1]))

        S = np.inner(X.T,X.T)
        N = len(X)
        mu = np.mean(X,axis=0)
        classLabels = np.unique(y)
        for label in classLabels:
            classIdx = np.argwhere(y==label).T[0]
            Nl = len(classIdx)
            xL = X[classIdx]
            muL = np.mean(xL,axis=0)
            muLbar = muL - mu
            Sb = Sb + Nl*np.outer(muLbar,muLbar)

        Sbar = S - N*np.outer(mu,mu)
        Sw = Sbar - Sb
        self.mean_ = mu

        return (Sw,Sb)
