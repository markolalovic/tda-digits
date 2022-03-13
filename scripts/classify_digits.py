#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' classify_digits.py: Performs the following steps:

   * Load the extracted topological features;
   * Classify the images using SVM with RBF kernel;
   * Evaluate the model using CV on train set and once on test set

'''

from __future__ import print_function # if you are using Python 2
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from tda_digits import *

n_samples = 10000
if len(sys.argv) == 2:
    print('Setting n_samples to: %i' % (n_samples))
    n_samples = int(sys.argv[1])

def grid_search(X, y, C_range, gamma_range):
    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedShuffleSplit(n_splits=5,
                                test_size=0.2,
                                random_state=42)

    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    values = grid.best_params_
    print('The best parameter values are %s with a score of %0.2f' %
        (values, grid.best_score_))

    return values

def show_misclassified(classifier, X_test, y_test, actual=2, mistaken_for=5):
    '''Shows (maximum 3) examples of misclassified image.'''
    n_examples = 0
    for n in range(n_samples // 2):
        if int(y_test[n]) == actual:
            features = X_test[n, ]
            features = np.array([features.transpose()])
            predicted = classifier.predict(features)[0]
            if predicted == str(mistaken_for):
                print('Image of %d was misclassified as: %s'
                    % (actual, predicted))
                nn = n_samples // 2 + n # to skip train set
                get_image(nn, show=True)

                n_examples += 1
                if n_examples == 3:
                    break

if __name__ == '__main__':
    ##
    # Load the extracted topological features
    X = np.load('../data/features_'+ str(n_samples) + '.npy', allow_pickle=True)
    y = np.load('../data/y_' + str(n_samples) + '.npy', allow_pickle=True)

    # we are cheating a bit in scaling all of the data, instead of fitting
    # the transformation on training set and applying it on test set
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split 50:50
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
    print('Number of samples in train set: %i' % (X_train.shape[0]) )
    print('Number of samples in test set: %i' % (X_test.shape[0]) )

    # perform the grid search
    if n_samples > 1000: # it takes some time
        # The best parameter values are
        # {'C': 138.94954943731375, 'gamma': 0.006551285568595509}
        # with a score of 0.89
        # Accuracy: 0.8872
        values = {'C': 138.94954943731375, 'gamma': 0.006551285568595509}
    else:
        C_range = np.logspace(0, 3, 50)
        gamma_range = np.logspace(-3, 1, 50)
        values = grid_search(X_train, y_train, C_range, gamma_range)


    ##
    # Evaluate the model

    ## Results of 10-fold CV results on train set
    classifier = svm.SVC(C=values['C'], gamma=values['gamma'])
    scores = cross_val_score(classifier, X_train, y_train, cv=10)
    print('Accuracy on train set: %0.2f (+/- %0.2f)' %
         (scores.mean(), scores.std() * 2))

    ## Results on test set
    classifier.fit(X_train, y_train) # train the model on train set
    predicted = classifier.predict(X_test) # predict on test set

    print('Accuracy on test set: %0.2f' %
          metrics.accuracy_score(y_test, predicted))

    print('Classification results on test set:\n%s' %
          metrics.classification_report(y_test, predicted))

    print('Confusion matrix on test set:\n%s' %
          metrics.confusion_matrix(y_test, predicted))
