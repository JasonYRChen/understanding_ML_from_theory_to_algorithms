import numpy as np
from pprint import pprint


def halfspaces(x: list[list[float]], y: list[int], intercept_on=True) ->\
    (list[float], float):
    # Check if inputs are instances of np.ndarray. If not, make a copy of
    # np.ndarray of them and issue a warning on copying.
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        print("Warning: the input 'x' is not a valid np.ndarray. " +\
              "A new np.ndarray will produce and extra memory are needed.")
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        print("Warning: the input 'y' is not a valid np.ndarray. " +\
              "A new np.ndarray will produce and extra memory are needed.")

    # Concatenate a 1's column at the right-most of x for intercept
    if intercept_on:
        x = np.hstack([x, np.ones(x.shape[0])[:, np.newaxis]])
    # Reshape y to n x 1 if it's not
    if y.shape[-1] != 1:
        y = y[:, np.newaxis]

    # Iterate and find coefficients
    w = np.zeros(x.shape[1])
    result = (x * y).dot(w)
    while np.any(result <= 0):
        index = np.argmax(result <= 0)
        w += y[index] * x[index]
        result = (x * y).dot(w)

    # Output according to whether intercept is included
    if intercept_on:
        new_w, intercept = np.delete(w, -1), w[-1]
        return new_w, intercept
    return w, 


if __name__ == '__main__':
    x1 = [[n / j for n in range(4, 6)] for j in range(1, 5)]
    x2 = [[3, 4], [1, 7], [-1, 3], [-4, -4]]
    y = [1, 1, -1, -1]
#    pprint(x)
    x = x1
    y = np.array(y)
    print(x)
    print(y)
    print(halfspaces(x, y, True))
