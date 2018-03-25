from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from sklearn.datasets import load_svmlight_file

from rrf import RRF

DATA_DIR = "example_data"
N_RUNS = 10

if __name__ == '__main__':

    print("=========== Test classification on cod-rna data ===========")
    # X in [-1, 1]
    X, y = load_svmlight_file(DATA_DIR + "/cod-rna.scale", n_features=8)
    X = X.toarray()

    score = np.zeros(N_RUNS)
    train_time = np.zeros(N_RUNS)
    test_time = np.zeros(N_RUNS)

    model = [None] * N_RUNS

    for r in range(N_RUNS):
        idx = np.random.permutation(X.shape[0])

        c = RRF(loss='hinge', task="classification", learning_rate=0.003,
                learning_rate_gamma=0.0001, gamma=1.0, D=100)

        c.fit(X[idx], y[idx])
        train_time[r] = c.train_time
        score[r] = c.mistake

        print("Mistake rate = %.4f" % score[r])
        print("Training time = %.4f" % train_time[r])
        model[r] = c

    print("%.4f\t%.3f+-%.3f" % (train_time.mean(), 100 * score.mean(), 100 * score.std()))

    print("=========== Test regression on cod-rna data ===========")
    # X in [-1, 1]
    X, y = load_svmlight_file(DATA_DIR + "/cod-rna.scale", n_features=8)
    X = X.toarray()

    score = np.zeros(N_RUNS)
    train_time = np.zeros(N_RUNS)
    test_time = np.zeros(N_RUNS)

    model = [None] * N_RUNS

    for r in range(N_RUNS):
        idx = np.random.permutation(X.shape[0])

        c = RRF(loss='l2', task="regression", learning_rate=0.003,
                learning_rate_gamma=0.0001, gamma=1.0, D=100)

        c.fit(X[idx], y[idx])
        train_time[r] = c.train_time
        score[r] = c.mistake

        print("Mistake rate = %.4f" % score[r])
        print("Training time = %.4f" % train_time[r])
        model[r] = c

    print("%.4f\t%.3f+-%.3f" % (train_time.mean(), 100 * score.mean(), 100 * score.std()))
