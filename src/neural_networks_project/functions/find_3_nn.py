from math import inf
import numpy as np

def Euclidean_distance(vec1, vec2):
    return np.sum((vec1 - vec2) ** 2) # Withought sqrt I am comparing distances ^ 2 to distances ^ 2 so closest 3 n withh still be the same

def Find_3_nn(test_data, data_batch, labels_batch):
    distances_labels = [[inf, -1], [inf, -1], [inf, -1]]
    for i in range(len(data_batch)):
        cur = distances_labels[2][0]
        pot = Euclidean_distance(test_data, data_batch[i])
        if pot < cur:
            distances_labels[2][0] = pot
            distances_labels[2][1] = labels_batch[i]
            distances_labels.sort(key=lambda x: x[0])
    return distances_labels

