import numpy as np
import matplotlib.pyplot as plt
from decision_stumps import decision_stump
from decision_stumps import decision_stump_classifier as dsc
from decision_stumps import decision_stump_multiclass as dsm
from decision_stumps import decision_stump_multiclass_classifier as dsmc
from decision_stumps import decision_stump_multiclass_gini_classifier as dsmgc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.tree import DecisionTreeClassifier as dtc


def adaboost_multiclass(X: np.ndarray, y: np.ndarray, cycles: int=50):
    """
        Ready to test this function. Starts with 2 features, then compare 2 more features simulation to sklearn.
    """

    # label and 0-base index mapping
    index_to_label = np.unique(y)
    label_to_index = {k: i for i, k in enumerate(index_to_label)}

    distributions = np.full(X.shape[0], 1/X.shape[0])

    classes = len(index_to_label)
    hypotheses = np.zeros((X.shape[0], classes))
    for i in range(cycles):
        hypothesis = dsmc(X, y, distributions)
#        hypothesis = dsmgc(X, y, distributions) # self-made gini version
#        error = sum(distributions[hypothesis != y]) # from textbook
        error = np.mean(distributions[hypothesis != y]) # from sklearn AdaBoost

        if not error: # perfect classification
            break

        weight = np.log(1/error - 1) + np.log(classes - 1)

        # renew distribution
        distributions[hypothesis != y] *= np.exp(weight)
        distributions /= sum(distributions)

        # renew hypotheses
        for i, l in enumerate(hypothesis):
            c = label_to_index[l]
            hypotheses[i, c] += weight

    y_hat = index_to_label[hypotheses.argmax(1)]
    return y_hat


def adaboost_function(data: np.ndarray, labels: np.ndarray, cycles: int):
    """
        Return classified labels by Adaboost via decision stump as base
        estimator.

        paras:
          data: np.ndarray, n x d matrix-like data to learn.
          labels: np.ndarray, n labels.
          cycles: int, iterations of boosting. It may terminate earlier if
            well classified.

        return:
          hypotheses: np.ndarray, n hypothesized labels.
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    distributions = np.full(data.shape[0], 1/data.shape[0])
    hypotheses = np.zeros(data.shape[0])
    for i in range(cycles):
        hypothesis = dsc(data, labels, distributions)
        error = sum(distributions[hypothesis != labels])
        if not error: # perfect classified, no need further learning
            break
        weight = np.log(1/error - 1) * 0.5
        hypotheses += weight * hypothesis
        distributions *= np.exp(-weight * labels * hypothesis)
        distributions /= sum(distributions)
    hypotheses = np.where(hypotheses > 0, 1, -1)
    return hypotheses


def adaboost_test(data: np.ndarray, labels: np.ndarray, cycles: int):
    """
        This is a test to another loss function, which depresses correct 
        classification to only 1 instead of a much smaller decimal, which
        could be useful to classify multi-class
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    distributions = np.full(data.shape[0], 1/data.shape[0])
    hypotheses = np.zeros(data.shape[0])
    for i in range(cycles):
        hypothesis = dsc(data, labels, distributions)
        error = sum(distributions[hypothesis != labels])
        if not error: # perfect classified, no need further learning
            break
        weight = np.log(1/error - 1) 
        distributions *= np.exp(weight * np.where(labels != hypothesis, 1, 0))
        distributions /= sum(distributions)
        hypotheses += weight * hypothesis 
    hypotheses = np.where(hypotheses > 0, 1, -1)
    return hypotheses


def adaboost_function_demo(data: np.ndarray, labels: np.ndarray, cycles: int):
    distributions = np.full(data.shape[0], 1/data.shape[0])
    hypotheses = np.zeros(data.shape[0])
    plt.figure('adaboost demo')
    for i in range(cycles):
        feature, stump, polarity = decision_stump(data, labels, distributions)
        xx, yy = ((-50, 50), (stump, stump)) if feature else ((stump, stump), (-50, 50))
        plt.plot(xx, yy, c='g')
        hypothesis = np.where(data[:, feature] < stump, polarity, -polarity)
        error = sum(distributions[hypothesis != labels])
        if not error: # perfect classified, no need further learning
            break
        weight = np.log(1/error - 1) * 0.5
        hypotheses += weight * hypothesis
        distributions *= np.exp(-weight * labels * hypothesis)
        distributions /= sum(distributions)
    hypotheses = np.where(hypotheses > 0, 1, -1)
    
    color = np.where(hypotheses > 0, 'b', 'r')
    plt.scatter(data[:, 0], data[:, 1], c=color)


# build an adaboost to absorb sklearn.tree.DecisionTreeClassifier
class adaboost:
    def __init__(self, estimator=None, n_estimators=50):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.hypotheses = None


    def fit(self, X, y):
        index_to_label = np.unique(y)
        label_to_index = {k: i for i, k in enumerate(index_to_label)}
        distributions = np.full(X.shape[0], 1/X.shape[0])
        classes = len(index_to_label)
        hypotheses = np.zeros((X.shape[0], classes))
        for i in range(cycles):
            clf = self.estimator(max_depth=1)
            clf.fit(X, y, distributions)
            hypothesis = clf.predict(X)
#            error = sum(distributions[hypothesis != y])
            error = np.mean(distributions[hypothesis != y])
            if not error:
                break
            weight = np.log(1/error - 1) + np.log(classes - 1)

            distributions[hypothesis != y] *= np.exp(weight)
            distributions /= sum(distributions)

            for i, l in enumerate(hypothesis):
                c = label_to_index[l]
                hypotheses[i, c] += weight

        self.hypotheses = index_to_label[hypotheses.argmax(1)]


    def predict(self, data):
        return self.hypotheses


if __name__ == '__main__':
    sign = 1 # 1 or -1
    cycles = 50
    samples = 200
    features = 2
    data = (np.random.rand(samples, features) - 0.5) * 100

    # different labels to choose from
#    labels = np.where(data[:, 0]**2/900+data[:, 1]**2/400 > 1, -sign, sign)
#    labels[(-5 <= data[:, 0]) & (data[:, 0] <= 5)] = -sign

#    labels = np.where((data[:, 0] > 5) & (data[:, 1] < -5), -sign, sign)

    # new labels with multi-class to test my adaboost multiclass (x direction)
#    labels = np.zeros(samples)
#    quantiles = np.quantile(data[:, 0], np.arange(0, 1, 0.2))
#    for i, q in enumerate(quantiles):
#        labels[data[:, 0] >= q] = i

    # new labels with multi-class to test my adaboost multiclass (y direction)
    labels = np.zeros(samples)
    quantiles = np.quantile(data[:, 1], np.arange(0, 1, 0.2))
    for i, q in enumerate(quantiles):
        labels[data[:, 1] >= q] = i

    # original data
    plt.figure('original data')
#    colors_original = np.where(labels > 0, 'b', 'r')
#    plt.scatter(data[:, 0], data[:, 1], c=colors_original)
    colors = labels * 30
    plt.scatter(data[:, 0], data[:, 1], c=colors)

    # adaboost by me
#    labels_guess = adaboost_function(data, labels, cycles)
#    print(f'Error from adaboost by me: {sum(labels != labels_guess)}')
#    colors_new = np.where(labels_guess > 0, 'b', 'r')
#    plt.figure('my ada')
#    plt.scatter(data[:, 0], data[:, 1], c=colors_new)

    # adaboost from sklearn
    boost = abc(n_estimators=cycles)
    boost.fit(data, labels)
    predict = boost.predict(data)
    print(f'Error from sklearn: {sum(labels != predict)}')
    colors_sklearn = predict * 30
    plt.figure('sklearn ada')
    plt.scatter(data[:, 0], data[:, 1], c=colors_sklearn)

    # adaboost multiclass
    labels_guess = adaboost_multiclass(data, labels, cycles)
    print(f'Error from adaboost multiclass: {sum(labels != labels_guess)}')
    colors_new = labels_guess * 30
    plt.figure('adaboost multiclass')
    plt.scatter(data[:, 0], data[:, 1], c=colors_new)

    # adaboost by me with DecisionTreeClassifier
#    boost = adaboost(dtc, n_estimators=cycles)
#    boost.fit(data, labels)
#    predict = boost.predict(data)
#    print(f'Error from mixed: {sum(labels != predict)}')
#    colors_sklearn = predict * 30
#    plt.figure('my ada + decision tree')
#    plt.scatter(data[:, 0], data[:, 1], c=colors_sklearn)
    
    # adaboost demo
#    adaboost_function_demo(data, labels, cycles)

    # adaboost test
#    labels_guess = adaboost_test(data, labels, cycles)
#    print(f'Error from adaboost test: {sum(labels != labels_guess)}')
#    colors_new = np.where(labels_guess > 0, 'b', 'r')
#    plt.figure('adaboost test')
#    plt.scatter(data[:, 0], data[:, 1], c=colors_new)


    plt.show()
