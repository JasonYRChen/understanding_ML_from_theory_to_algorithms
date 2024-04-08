import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def decision_stumps(data: list[list[float]], labels: list[int], 
        distributions: list[float], positive_first: bool = True):
    """
        Find the minimum loss on every dimension of data and return the
        dimension and position of the stump and the corresponding loss.

        paras:
          data: n x d array-like, the data to learn.
          labels: n x 1 array of 1 or -1, specifying the category of 
            corresponding data.
          distributions: n x 1 array of [0, 1] floats, pointing out the
            probability of each data.
          positive_first: bool, True if it's 1 in labels to find first,
            False otherwise.

        returns:
          best_dim: int, the dimension to place stump.
          best_stump: float, the coordinate along best_dim.
          best_loss: float, the minimum loss along the learning.
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(distributions, np.ndarray):
        distributions = np.array(distributions)

    # how many tokens should visit on each dimension
    tokens_to_visit = sum(labels > 0 if positive_first else labels < 0)
    # the basic loss of not finding all the tokens before learning
    loss = sum(distributions[labels > 0 if positive_first else labels < 0])
    best_loss, best_dim, best_stump = float('inf'), 0, 0
    rows, cols = data.shape
    sign = 1 if positive_first else -1
    for col in range(cols):
        remain_tokens = tokens_to_visit
        remain_loss = loss
        sorted_arg = data[:, col].argsort()
        prev_pos = data[sorted_arg[0], col] - 1
        for arg in sorted_arg:
            # check loss and remained tokens from last loop
            current_pos = data[arg, col]
            if remain_loss < best_loss:
                best_loss = remain_loss
                best_dim = col
                best_stump = (prev_pos + current_pos) / 2
            if not remain_tokens:
                # all the tokens are found in this dimension. Further search
                # could only increase loss, so stop here
                break

            # renew loss and remained tokens with latest label
            label = labels[arg]
            odd = distributions[arg]
            remain_loss -= sign * label / abs(label) * odd
            if sign * label > 0:
                remain_tokens -= 1
            prev_pos = current_pos
        else:
            # this can only happen if the wanted token is at the very last
            # position in 'sorted_arg'. Another check on loss is then needed.
            current_pos = data[arg, col] + 1
            if remain_loss < best_loss:
                best_loss = remain_loss
                best_dim = col
                best_stump = (prev_pos + current_pos) / 2
    return best_dim, best_stump, best_loss


def decision_stumps_classifier(data: list[list[float]], labels: list[int], 
        distributions: list[float], positive_first: bool = True):

    """
        Output classified labels with decision stump algorithm.

        paras:
          data: n x d array-like, the data to learn.
          labels: n x 1 array of 1 or -1, specifying the category of 
            corresponding data.
          distributions: n x 1 array of [0, 1] floats, pointing out the
            probability of each data.
          positive_first: bool, True if it's 1 in labels to find first,
            False otherwise.

        return:
          new_labels: n x 1 array of 1 and -1, a hypothesis of the labels
            of data.
    """

    sign = 1 if positive_first else -1
    dim, stump, _ = decision_stumps(data, labels, distributions, positive_first)
    new_labels = np.where(data[:, dim] < stump, sign, -sign)
    return new_labels

    
if __name__ == '__main__':
    sign = 1 # 1 or -1
#    data = (np.random.rand(100, 5) - 0.5) * 100
    data = (np.random.rand(100, 2) - 0.5) * 100
    # spread in x direction
    data[:, 0].sort()
    labels = np.where(data[:, 0] > -1, -1 * sign, 1 * sign)
    # spread in y direction
#    data[:, 1].sort()
#    labels = np.where(data[:, 1] > 1, -1 * sign, 1 * sign) # y-direction

    np.random.shuffle(labels[40:60])
    distributions = np.random.rand(data.shape[0])
#    distributions = np.full(data.shape[0], 1/data.shape[0])
    dim, stump, loss = decision_stumps(data, labels, distributions,True if sign > 0 else False )
    print(f'Split dimension: {dim}, stump position: {stump}, loss: {loss}')

    colors = np.where(labels > 0, 'b', 'r')
    sizes = distributions * 100
#    f = plt.figure(1)
#    plt.scatter(data[:, 0], data[:, 1])
    g = plt.figure(1)
    g.suptitle('Original data with stump')
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=sizes)
    if dim:
        stump_plot = np.array([[min(data[:, 0]), stump], [max(data[:, 0]), stump]])
        labels_guess = np.where(data[:, 1] > stump, -1 * sign, 1 * sign)
    else:
        stump_plot = np.array([[stump, min(data[:, 1])], [stump, max(data[:, 1])]])
        labels_guess = np.where(data[:, 0] > stump, -1 * sign, 1 * sign)
    plt.plot(stump_plot[:, 0], stump_plot[:, 1], c='g')
    h = plt.figure(2)
    h.suptitle('Classified data by stump')
    colors_new = np.where(labels_guess > 0, 'b', 'r')
    plt.scatter(data[:, 0], data[:, 1], c=colors_new, s=sizes)
    plt.plot(stump_plot[:, 0], stump_plot[:, 1], c='g')

    plt.figure()
    color_guess = np.where(decision_stumps_classifier(data, labels, distributions, True if sign > 0 else False) > 0, 'b', 'r')
    plt.scatter(data[:, 0], data[:, 1], c=color_guess, s=sizes)
    plt.show()
