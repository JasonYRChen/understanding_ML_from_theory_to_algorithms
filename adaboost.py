import numpy as np
import matplotlib.pyplot as plt
from decision_stumps import decision_stump
from decision_stumps import decision_stump_classifier
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.tree import DecisionTreeClassifier as dtc


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
        hypothesis = decision_stump_classifier(data, labels, distributions)
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
        hypothesis = decision_stump_classifier(data, labels, distributions)
        error = sum(distributions[hypothesis != labels])
        if not error: # perfect classified, no need further learning
            break
#        weight = np.log(1/error - 1) * 0.5
#        hypotheses += weight * hypothesis 
#        distributions *= np.exp(weight * np.where(labels != hypothesis, 1, -1))
        weight = np.log(1/error - 1) 
        hypotheses += weight * hypothesis 
        distributions *= np.exp(weight * np.where(labels != hypothesis, 1, 0))
        distributions /= sum(distributions)
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

    def fit(self, data, labels):
        if self.hypotheses is None:
            self.hypotheses = np.zeros(data.shape[0])

        weights = np.full(data.shape[0], 1/data.shape[0])
#        print('adaboost with dtc')
        for i in range(self.n_estimators):
            clf = self.estimator(max_depth=1)
            clf.fit(data, labels, weights)
            hypothesis = clf.predict(data)
            error = sum(weights[hypothesis != labels])
            if not error:
                break
            weight = np.log(1/error - 1) * 0.5
            self.hypotheses += weight * hypothesis
#            print(f'error: {error}, weight: {weight}')
#            print('distribution  hypothesis')
#            print(np.vstack((weights, hypothesis)).T)
            weights *= np.exp(-weight * labels * hypothesis)
            weights /= sum(weights)
        self.hypotheses = np.where(self.hypotheses > 0, 1, -1)


    def predict(self, data):
        return self.hypotheses


if __name__ == '__main__':
    sign = 1 # 1 or -1
    cycles = 50 
    samples = 500
    features = 2
    data = (np.random.rand(samples, features) - 0.5) * 100

    # different labels to choose from
    labels = np.where(data[:, 0]**2/900+data[:, 1]**2/400 > 1, -sign, sign)
    labels[(-5 <= data[:, 0]) & (data[:, 0] <= 5)] = -sign
#    labels = np.where((data[:, 0] > 5) & (data[:, 1] < -5), -sign, sign)

    # original data
    colors_original = np.where(labels > 0, 'b', 'r')
    plt.figure('original data')
    plt.scatter(data[:, 0], data[:, 1], c=colors_original)

    # adaboost by me
    labels_guess = adaboost_function(data, labels, cycles)
    print(f'Error from adaboost by me: {sum(labels != labels_guess)}')
    colors_new = np.where(labels_guess > 0, 'b', 'r')
    plt.figure('my ada')
    plt.scatter(data[:, 0], data[:, 1], c=colors_new)

    # adaboost from sklearn
    boost = abc(n_estimators=cycles)
    boost.fit(data, labels)
    predict = boost.predict(data)
    print(f'Error from sklearn: {sum(labels != predict)}')
    colors_sklearn = np.where(predict > 0, 'b', 'r')
    plt.figure('sklearn ada')
    plt.scatter(data[:, 0], data[:, 1], c=colors_sklearn)

    # adaboost by me with DecisionTreeClassifier
    boost = adaboost(estimator=dtc, n_estimators=cycles)
    boost.fit(data, labels)
    predict = boost.predict(data)
    print(f'Error from mixed: {sum(labels != predict)}')
    colors_sklearn = np.where(predict > 0, 'b', 'r')
    plt.figure('my ada + decision tree')
    plt.scatter(data[:, 0], data[:, 1], c=colors_sklearn)
    
    # adaboost demo
    adaboost_function_demo(data, labels, cycles)

    # adaboost test
    labels_guess = adaboost_test(data, labels, cycles)
    print(f'Error from adaboost test: {sum(labels != labels_guess)}')
    colors_new = np.where(labels_guess > 0, 'b', 'r')
    plt.figure('adaboost test')
    plt.scatter(data[:, 0], data[:, 1], c=colors_new)


    plt.show()
