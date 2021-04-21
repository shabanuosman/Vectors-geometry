import numpy as np
import random


def create_array_from_function(func, d_shape, dtype=None):
    """
    Creates an  array from the given lambda and dimensions shape
    
    :param func: lambda function to populate the values at each row and column indices
    :param d_shape: dimensions shape
    :param dtype: array data type
    :return: an array 
    """
    return np.fromfunction(func, d_shape, dtype=dtype)


def boundary_cropping(arr, mask):
    """
    
    :param arr: an array
    :param mask: a binary mask array
    :return: a boundary cropped array produced by masking the given array with the masking array
    """
    non_zero_indices = np.argwhere(mask)
    print(non_zero_indices)
    begin_nd_corners = non_zero_indices.min(axis=0)
    finish_nd_corners = non_zero_indices.max(axis=0) + 1
    ndslice = tuple(slice(begin, finish) for (begin, finish) in zip(begin_nd_corners, finish_nd_corners))
    return arr[ndslice]


def shuffle_list_inplace_constant_memory(l):
    """
     Shuffles a given list
    :param l: a list
    :return: a shuffled list
    """
    random.seed(0)
    random.shuffle(l)
    return l


def pop_var_from_subpop_var(groups):
    """
     calculating the variance of the entire population from the variance of the subpopulations.
    :param groups: groups
    :return: 
    """
    pop_mean = (1 + 2 + 3 + 4 + 5 + 6) / 6
    pop_var = (
                  sum([group.size * ((group.mean() - pop_mean) ** 2) for group in groups]) + \
                  sum([(group.size - 1) * group.var(ddof=1) for group in groups])
              ) / sum([group.size for group in groups])
    return pop_var


def shape_as_blocks(arr, nrows, ncols):
    """
   
   :param arr: 
   :param nrows: 
   :param ncols: 
   :return: 
   """
    height, width = arr.shape

    return (arr.reshape(height // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


# inputs

## Functional Arrays
print("Functional Arrays")
print(create_array_from_function(lambda i, j: (i - j) ** 2, [4, 4]))
print("-----------------------------")


## Removing Boundaries
print('Removing Boundaries')
a1 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]])
a2 = np.array([[[0, 0, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
print(boundary_cropping(a1, a1 != 0))
print(boundary_cropping(a2, a2 != 0))
print("-----------------------------")

## Block Reshaping
print('Block Reshaping')
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]])
print(shape_as_blocks(arr, 2, 2))
print("-----------------------------")


## Population Variance from Subpopulation Variance
print('Population Variance from Subpopulation Variance')
groups = [np.array([1, 2, 3, 4]), np.array([5, 6])]
print(pop_var_from_subpop_var(groups))
print("-----------------------------")


## Shuffle a Large List
print('Shuffle a Large List')
print(shuffle_list_inplace_constant_memory([1, 2, 3, 4, 5]))
print("-----------------------------")

