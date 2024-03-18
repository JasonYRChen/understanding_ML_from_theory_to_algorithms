import numpy as np
from pprint import pprint


def halfspaces(x: list[list[float]], y: list[int], intercept_on=True) ->\
    (list[float], float):
    """
        ****************************WARNING******************************
        * This version is not capable of dealing with non-seperable     *
        * dataset. If a non-seperable one is given, this may cause the  *
        * the program into infinite loop.                               *
        *****************************************************************

        Return the linear combination coefficients of the hyperplane of a
        halfspace that divides the data into two categories. The returned
        coefficients may or may not contain intercept, or constant part,
        should the 'intercept_on' is True or False.

        paras:
          x: n*m list-like or np.ndarray. Seperable data to learn , each 
             row is the coordinate of the corresponding datum.
          y: list or np.ndarray of n int in {-1, +1}. Each of which is
             the label of the datum in the corresponding index of x.
          intercept_on: bool, used to control to find the intercept or
             not. It is recommanded to stay True unless it is sure to be
             unnecessary.

        return:
          Tuple of coefficients. The first element in the tuple is the
             coefficients without intercept, and the second element is
             the intercept. Should 'intercept_on' is False, the second
             element will not return.
    """

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
#        print('*********************************************')
#        print(f'old coefficients: {w}')
#        print(f'old result: {result}')
        index = np.argmax(result <= 0)
        w += y[index] * x[index]
        result = (x * y).dot(w)
#        print(f'first element to be non-positive: {index}')
#        print(f'new coefficients: {w}')
#        print(f'new result: {result}')

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
    x = np.array(x)
    y = np.array(y)
    print('x\n', x)
    print('y\n', y)
    print(halfspaces(x, y, True))
