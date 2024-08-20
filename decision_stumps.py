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
            if min(loss_p, loss_n) < loss_best:
                best_feature = col
                best_stump = (nodes[i] + nodes[i+1]) / 2
                loss_best = min(loss_p, loss_n)
                polarity = 1 if loss_p < loss_n else -1
                
            # calculate loss_pos and loss_neg
            if i < rows:
                index = indices[i]
                loss_p -= y[index] * sample_weight[index]
                loss_n += y[index] * sample_weight[index]
    return best_feature, best_stump, polarity


def decision_stump_multiclass_gini(X: np.ndarray, y: np.ndarray, 
        sample_weight: np.ndarray=None):

    rows, cols = X.shape
    if sample_weight is None:
        sample_weight = np.ones(rows) / rows

    # translate labels in y to 0-base integers
    index_to_y = np.unique(y)
    y_to_index = {k: i for i, k in enumerate(index_to_y)} # reverse translation
    y_int = np.empty(len(y), dtype=int) # map y to integers
    for i, label in enumerate(index_to_y):
        y_int[y == label] = i

    # initial loss on each side
    loss_l, loss_r = np.zeros(len(index_to_y)), np.empty(len(index_to_y))
    for i in range(len(loss_r)):
        loss_r[i] = sum(sample_weight[y_int == i])

    feature, stump, label_l, label_r = None, None, None, None # initialization
    loss_best = np.inf 
    nodes = np.empty(rows + 2)
    for col in range(cols):
        loss_l_copy, loss_r_copy = loss_l.copy(), loss_r.copy()
        n_l, n_r, n_total = 0, len(y), len(y)
        indices = np.argsort(X[:, col])
        nodes[1:-1] = X[indices, col]
        nodes[0], nodes[-1] = nodes[1] - 1, nodes[-2] + 1
        for i in range(rows + 1):
            # renew loss_best
            loss_l_sum, loss_r_sum = sum(loss_l_copy), sum(loss_r_copy)
            p_l = loss_l_copy / loss_l_sum if loss_l_sum else np.zeros(len(index_to_y))
            p_r = loss_r_copy / loss_r_sum if loss_r_sum else np.zeros(len(index_to_y))
            loss = n_l/n_total * (p_l @ (1-p_l)) + n_r/n_total * (p_r @ (1-p_r))
            if loss < loss_best:
                feature = col
                stump = (nodes[i] + nodes[i+1]) / 2
                loss_best = loss
                label_l = index_to_y[p_l.argmax()]
                label_r = index_to_y[p_r.argmax()]

            # recalculate loss
            if i < rows:
                index = indices[i]
                n_l += 1
                n_r -= 1
                loss_l_copy[y_int[index]] += sample_weight[index]
                loss_r_copy[y_int[index]] -= sample_weight[index]
    return feature, stump, label_l, label_r


def decision_stump_multiclass(X: np.ndarray, y: np.ndarray, 
        sample_weight: np.ndarray=None):
    """
        Support non-integer labeling.
    """

    rows, cols = X.shape
    if sample_weight is None:
        sample_weight = np.ones(rows) / rows

    # translate labels in y to 0-base integers
    index_to_y = np.unique(y)
    y_to_index = {k: i for i, k in enumerate(index_to_y)} # reverse translation
    y_int = np.empty(len(y)) # map y to integers
    for i, label in enumerate(index_to_y):
        y_int[y == label] = i

    # loss of each y element at the beginning
    losses = np.zeros(len(index_to_y))
    for i in range(len(losses)):
        losses[i] = sum(sample_weight[y_int == i])

    feature, stump, label = None, None, None # initialization
    loss_best = np.inf 
    nodes = np.empty(rows + 2)
    for col in range(cols):
        losses_copy = losses.copy()
        indices = np.argsort(X[:, col])
        nodes[1:-1] = X[indices, col]
        nodes[0], nodes[-1] = nodes[1] - 1, nodes[-2] + 1
        for i in range(rows + 1):
            # renew loss_best
            index_min = losses_copy.argmin()
            if losses_copy[index_min] < loss_best:
                feature = col
                stump = (nodes[i] + nodes[i+1]) / 2
                loss_best = losses_copy[index_min]
                label = index_to_y[index_min]

            # recalculate loss
            if i < rows:
                index = indices[i]
                losses_copy += sample_weight[index]
                losses_copy[y_to_index[y[index]]] -= sample_weight[index] * 2

    # find label_oppos
    label_oppos = label
    max_weights_sum = 0
    weights_selected = sample_weight[X[:, feature] > stump]
    y_selected = y[X[:, feature] > stump]
    for i in np.unique(y_selected):
        weights_sum = sum(weights_selected[y_selected == i])
        if weights_sum > max_weights_sum:
            label_oppos, max_weights_sum = i, weights_sum

    return feature, stump, label, label_oppos


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


def decision_stump_multiclass_classifier(X: np.ndarray, y: np.ndarray, 
        sample_weight: np.ndarray=None):
    feature, stump, label, label_oppos = decision_stump_multiclass(X, y, sample_weight)
    y_hat = np.where(X[:, feature] < stump, label, label_oppos)
    return y_hat


def decision_stump_multiclass_gini_classifier(X: np.ndarray, y: np.ndarray, 
        sample_weight: np.ndarray=None):
    feature, stump, label, label_oppos = decision_stump_multiclass_gini(X, y, sample_weight)
    y_hat = np.where(X[:, feature] < stump, label, label_oppos)
    return y_hat
    

if __name__ == '__main__':
    sign = 1 # 1 or -1
    data_number = 500
    diameter = 100

    data = (np.random.rand(data_number, 2) - 0.5) * 100

    # spread in x direction
#    data[:, 0].sort()
#    labels = np.where(data[:, 0] > -10, -1 * sign, 1 * sign)
    # spread in y direction
#    data[:, 1].sort()
#    labels = np.where(data[:, 1] > 1, -sign, sign) # y-direction

    # in 4th quadrant
    labels = np.where((data[:, 0] > 5) & (data[:, 1] < -5), -sign, sign)

    # mess with partial labels
#    np.random.shuffle(labels[data_number//2-diameter:data_number//2+diameter])

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
#    f, s, p = decision_stump(data, labels, distributions)
#    y = np.where(data[:, f] < s, p, -p)
#    plt.figure('decision stump')
#    colors = np.where(y > 0, 'b', 'r')
#    plt.scatter(data[:, 0], data[:, 1], c=colors, s=distributions*100)
#    xx, yy = ([-50, 50], [s, s]) if f else ([s, s], [-50, 50])
#    plt.plot(xx, yy, 'g')

    # decision stump multiclass classifier
    y = decision_stump_multiclass_classifier(data, labels, distributions)
    colors = np.where(y > 0, 'b', 'r')
    plt.figure('multiclass classifier')
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=distributions*100)

    # decision stump multiclass gini classifier
    y = decision_stump_multiclass_gini_classifier(data, labels, distributions)
    colors = np.where(y > 0, 'b', 'r')
    plt.figure('multiclass classifier (gini)')
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=distributions*100)


    plt.show()
