from csv import reader
from math import sqrt

from random import seed, randrange

# Data Preprocessing


def load_csv(filename):
    dataset = list()
    with open(filename, "r") as file:
        lines = reader(file)
        for row in lines:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_col_to_float(dataset, column):
    for row in dataset:
        row[column] = float(str(row[column]).strip())
    return dataset


def str_col_to_int(dataset, column):
    class_values = [ row[column] for row in dataset]
    unique_values = set(class_values)

    lookup = dict()
    for i, value in enumerate(unique_values):
        lookup[value] = i

    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup



# Scaling data

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [ row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Normalize dataset
# scaled_value = (value - min)/ (max - min)
# rescaling an input variable to the range between 0 and 1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1] - minmax[i][0])


def column_mean(dataset):
    means = [ 0 for i in range(len(dataset[0])) ]
    for i in range(len(dataset[0])):
        col_values = [ row[i] for row in dataset ]
        means[i] = sum(col_values)/len(col_values)

    return means

# The standard deviation describes the average spread of values from the mean. It can be
# calculated as the square root of the sum of the squared di erence between each value and the
# mean and dividing by the number of values minus 1.
# standard deviation = square_root((Summation i=1 to N ( value_i - mean )^2)/count(values)-1)

def column_stdevs(dataset, means):
    stdevs = [ 0 for i in range(len(dataset[0])) ]
    for i in range(len(dataset[0])):
        variance = [ pow(row[i] - means[i], 2) for row in dataset ]
        stdevs[i] = sqrt(sum(variance)/(len(variance)-1))

    return stdevs

# standardize_value = (value - mean)/ std
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = ( row[i] - means[i] )/stdevs[i]


def train_test_split(dataset, split=0.60):
    train = list()
    train_size = split * len(dataset)
    # print(len(dataset))
    dataset_copy = list(dataset.values)
    print(len(dataset_copy))
    while len(train)< train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# The k-fold cross-validation method (also called just cross-validation) is
# a resampling method that provides a more accurate estimate of algorithm performance.
# It does this by rst splitting the data into k groups. The algorithm is then trained
# and evaluated k times and the performance summarized by taking the mean performance
# score.Each group of data is called a fold, hence the name k-fold cross-validation.

def cross_validation(data, folds):
    dataset_split = list()
    dataset_copy = list(data)
    fold_size = int(len(dataset_copy)/folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
    