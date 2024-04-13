import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def decision_stump(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray=None):
    """
        Return the optimal choice of feature, stump and polarity to classify
        the data X with sample weight by decision stump.

        paras:
          X: np.ndarray, n x d matrix-like data with n elements and d features.
          y: np.ndarray, n labels of {1, -1}.
          sample_weight: np.ndarray, n sample weights. If None is provided, an
            uniform distribution is implemented.

        returns:
          best_feature: int, the optimal choice of feature which is in [0, d).
          best_stump: float, the stump to divide the data in two groups.
          polarity: int, to classify the two groups which is either {1, -1}.
    """

    rows, cols = X.shape
    if sample_weight is None:
        sample_weight = np.ones(rows)

    best_feature, best_stump, polarity = None, None, 1 # initialization
    loss_best = np.inf 
    loss_pos = sum(sample_weight[y == 1])
    loss_neg = sum(sample_weight[y == -1])
    nodes = np.empty(rows + 2)
    for col in range(cols):
        loss_p, loss_n = loss_pos, loss_neg
        indices = np.argsort(X[:, col])
        nodes[1:-1] = X[indices, col]
        nodes[0], nodes[-1] = nodes[1] - 1, nodes[-2] + 1
        for i in range(rows + 1):
            # renew loss_best
            if min(loss_pos, loss_neg) < loss_best:
                best_feature = col
                best_stump = (nodes[i] + nodes[i+1]) / 2
                loss_best = min(loss_pos, loss_neg)
                polarity = 1 if loss_pos < loss_neg else -1
                
            # calculate loss_pos and loss_neg
            if i < rows:
                index = indices[i]
                loss_pos -= y[index] * sample_weight[index]
                loss_neg += y[index] * sample_weight[index]
    return best_feature, best_stump, polarity


def decision_stump_classifier(X: np.ndarray, y: np.ndarray, 
        sample_weight: np.ndarray=None):
    """
        Return the optimal classification by decision stump.

        paras:
          X: np.ndarray, n x d matrix-like data with n elements and d features.
          y: np.ndarray, n labels of {1, -1}.
          sample_weight: np.ndarray, n sample weights. If None is provided, an
            uniform distribution is implemented.

        returns:
          y_hat: np.ndarray, n classified labels in {1, -1} by decision stump.
    """

    feature, stump, polarity = decision_stump(X, y, sample_weight)
    y_hat = np.where(X[:, feature] < stump, polarity, -polarity)
    return y_hat

    
if __name__ == '__main__':
    sign = 1 # 1 or -1
    data_number = 100
    diameter = 20

    data = (np.random.rand(data_number, 2) - 0.5) * 100
    # spread in x direction
    data[:, 0].sort()
    labels = np.where(data[:, 0] > -10, -1 * sign, 1 * sign)
    # spread in y direction
#    data[:, 1].sort()
#    labels = np.where(data[:, 1] > 1, -sign, sign) # y-direction

    # mess with partial labels
    np.random.shuffle(labels[data_number//2-diameter:data_number//2+diameter])

    distributions = np.random.rand(data.shape[0])
#    distributions = np.full(data.shape[0], 1/data.shape[0])
    
    # original data
    plt.figure('original')
    colors = np.where(labels > 0, 'b', 'r')
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=distributions*100)

    # decision stump classifier
    y = decision_stump_classifier(data, labels, distributions)
    colors = np.where(y > 0, 'b', 'r')
    plt.figure('decision stump classifer')
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=distributions*100)

    # decision stump
    f, s, p = decision_stump(data, labels, distributions)
    y = np.where(data[:, f] < s, p, -p)
    plt.figure('decision stump')
    colors = np.where(y > 0, 'b', 'r')
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=distributions*100)
    xx, yy = ([-50, 50], [s, s]) if f else ([s, s], [-50, 50])
    plt.plot(xx, yy, 'g')


    plt.show()
