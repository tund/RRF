from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import numpy as np

np.seterr(all='raise')

from sklearn.utils import check_random_state

INF = 1e+8


class RRF(object):
    """Reparameterized Random Features
    """

    def __init__(self,
                 D=100,  # number of random features
                 lbd=1.0,
                 eps=0.1,
                 gamma=0.1,
                 loss='hinge',
                 task='classification',
                 num_epochs=10,
                 learning_rate=0.005,
                 learning_rate_gamma=0.005,
                 random_state=None,
                 verbose=0):
        self.D = D
        self.eps = eps
        self.gamma = gamma  # kernel width
        self.lbd = lbd  # regularization parameter
        self.loss = loss
        self.task = task
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_gamma = learning_rate_gamma
        self.random_state = random_state
        self.verbose = verbose

    def _init(self):
        self.epoch = 0
        self.history = None
        self.num_classes = 0
        self.label_encoder = None
        self.start_time = 0.0
        self.train_time = 0.0
        self.stop_training = 0
        self.exception = False
        self.best = -INF
        self.best_params = None
        self.random_engine = check_random_state(self.random_state)

        self.w = None
        self.sv = None  # support vectors
        if self.loss == 'hinge' or self.loss == 'logit':
            self.task = 'classification'
        else:
            self.task = 'regression'

        self.omega = None
        self.mistake = INF

        self.gamma_ = None
        self.e = None
        self.num_features = 0  # number of data features

    def _init_params(self, x):
        if self.num_classes > 2:
            self.w = 0.01 * self.random_engine.randn(2 * self.D, self.num_classes)
        else:
            self.w = 0.01 * self.random_engine.randn(2 * self.D)
        self.omega = self.gamma * self.random_engine.randn(x.shape[1], self.D)
        self.num_features = x.shape[1]
        self.gamma_ = np.log(self.gamma) * np.ones(self.num_features)  # Nx1
        self.e = self.random_engine.randn(self.num_features, self.D)  # NxD (\epsilon ~ N(0, 1))

    def fit(self, x, y):
        """Fit the model to the data x and the label y
        """

        # copy to avoid modifying
        x = x.copy()
        y = y.copy()

        if x.ndim < 2:
            x = x[..., np.newaxis]

        self._init()
        self._init_params(x)

        self.start_time = time.time()

        mistake = 0.0
        for t in range(x.shape[0]):
            phi = self._get_phi(x[[t]])

            wx = phi.dot(self.w)  # (x,)
            if self.task == 'classification':
                if self.num_classes == 2:
                    y_pred = np.uint8(wx >= 0)
                else:
                    y_pred = np.argmax(wx)
                mistake += (y_pred != y[t])
            else:
                mistake += (wx[0] - y[t]) ** 2
            dw, dgamma = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)  # compute gradients

            # update parameters
            self.w -= self.learning_rate * dw
            self.gamma_ -= self.learning_rate_gamma * dgamma

        self.mistake = mistake / x.shape[0]

        self.train_time = time.time() - self.start_time

        return self

    def _get_wxy(self, wx, y):
        m = len(y)  # batch size
        idx = range(m)
        mask = np.ones([m, self.num_classes], np.bool)
        mask[idx, y] = False
        z = np.argmax(wx[mask].reshape([m, self.num_classes - 1]), axis=1)
        z += (z >= y)
        return wx[idx, y] - wx[idx, z], z

    def _get_phi(self, x, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_
        omega = np.exp(gamma)[:, np.newaxis] * self.e  # NxD

        phi = np.zeros([x.shape[0], 2 * self.D])  # Mx2D
        xo = x.dot(omega)
        phi[:, :self.D] = np.cos(xo)
        phi[:, self.D:] = np.sin(xo)
        return phi

    def get_gamma_grad(self, x, phi, dphi, *args, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (N,)

        m = x.shape[0]  # batch size
        # gradient of \phi w.r.t \omega
        dpo = np.zeros([m, 2 * self.D, self.num_features])  # (M,2D,N)
        coswx, sinwx = phi[:, :self.D], phi[:, self.D:]  # (M,D)

        # broadcasting
        # drw[:, :self.D, :] = -X[:, np.newaxis, :] * sinwx[:, :, np.newaxis] * self.eps_.T[np.newaxis, :, :]  # (M,D,N)
        # drw[:, self.D:, :] = X[:, np.newaxis, :] * coswx[:, :, np.newaxis] * self.eps_.T[np.newaxis, :, :]  # (M,D,N)
        # dlg = drw.reshape([M * 2 * self.D, self.N_]).T.dot(dr.reshape(M * 2 * self.D)) * np.exp(g)

        # einsum
        dpo[:, :self.D, :] = np.einsum("mn,md,nd->mdn", -x, sinwx, self.e)  # (M,D,N)
        dpo[:, self.D:, :] = np.einsum("mn,md,nd->mdn", x, coswx, self.e)  # (M,D,N)
        dlg = np.einsum("mdn,md->n", dpo, dphi) * np.exp(gamma)

        return dlg

    def get_grad(self, x, y, *args, **kwargs):
        m = x.shape[0]  # batch size
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (N,)
        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (M,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (M,C)

        dw = self.lbd * w  # (2D,C)
        dgamma = np.zeros(gamma.shape)  # (N,)
        if self.num_classes > 2:
            wxy, z = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (M,2D)
                dphi = -w[:, y[wxy < 1]].T + w[:, z[wxy < 1]].T  # (M,2D)
                dgamma += self.get_gamma_grad(x[wxy < 1], phi[wxy < 1], dphi, gamma=gamma) / m
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, z].T)  # (M,2D)
                dgamma += self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / m
                dw[:, i] += d[z == i].sum(axis=0) / m
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / m
                dphi = -y[wxy < 1, np.newaxis] * w  # (M,2D)
                dgamma += self.get_gamma_grad(x[wxy < 1], phi[wxy < 1], dphi, gamma=gamma) / m
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (M,2D)
                dgamma = self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (M,2D)
                dgamma = self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (M,2D)
                dgamma += self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / m
                dphi = c[wxy] * w  # (M,2D)
                dgamma += self.get_gamma_grad(x[wxy], phi[wxy], dphi, gamma=gamma) / m
        return dw, dgamma
