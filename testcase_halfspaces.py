import numpy as np
from halfspaces import halfspaces
from pprint import pprint


def seperable_data_generation(coef, intercept, data_number):
    """
        Given coefficients and intercept of a linear equation, return the
        given number of seperable dataset and their bipartite label 
        counterparts.
    """

    coef = np.array(coef)
    data = (np.random.rand(data_number, len(coef)) - 0.5) * 10
    y = data.dot(coef) + intercept
    while np.any(y == 0): # data on the linear line is never allowed
        index = np.argmax(y == 0)
        data[index] = (np.random.rand(len(coef)) - 0.5) * 10
        y[index] = data[index].dot(coef) + intercept
    y = np.where(y > 0, 1, -1)
    return data, y


coef = [3, 4, 5]
intercept = -2
data_number = 2000 
data, label = seperable_data_generation(coef, intercept, data_number)
print(f'training data:\n{data}\nlabel:\n{label}\n')
print(f'original coefficients: {coef}, intercept: {intercept}')
coef_guess, itcpt = halfspaces(data, label)
print(f'guess coefficients: {coef_guess}\nguess intercept: {itcpt}')
