from functions.load_data import Load_data_Cifar10
from math import inf
import numpy as np
import time

def NN():

    _time = time.time()
    #Getiing all data
    (
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
    ) = Load_data_Cifar10()
   
    log_nn = [] 
    log_nn3 = []
    repetitions = len(data_train)
    for i in range(repetitions):
        __time = time.time()
        nn_1 = Find_3_nn(data_train[i], data_batch_1, labels_batch_1)
        nn_2 = Find_3_nn(data_train[i], data_batch_2, labels_batch_2)
        nn_3 = Find_3_nn(data_train[i], data_batch_3, labels_batch_3)
        nn_4 = Find_3_nn(data_train[i], data_batch_4, labels_batch_4)
        nn_5 = Find_3_nn(data_train[i], data_batch_5, labels_batch_5)
        all_nn = nn_1 + nn_2 + nn_3 + nn_4 + nn_5
        all_nn.sort(key=lambda x:x[0])
    
        __time = time.time() - __time 
        if all_nn[1][1] == all_nn[2][1]:
            nnof = all_nn[1][1]
        else:
            nnof = all_nn[0][1]
        log_nn3.append(nnof == labels_train[i])
        log_nn.append(all_nn[0][1] == labels_train[i])
        print(f'Image {i} {__time:.2f}s, 1nn {log_nn[i]} 3nn {log_nn3[i]}' )
    
    nn3stats = np.mean(log_nn3)
    nnstats = np.mean(log_nn)
    print(time.time() - _time)
    return nn3stats,nnstats, time.time() - _time, log_nn, log_nn3

def Find_3_nn(test_data, data_batch, labels_batch):
    distances_labels = [[inf, -1], [inf, -1], [inf, -1]]
    for i in range(len(data_batch)):
        cur = distances_labels[2][0]
        pot = np.sum((test_data - data_batch[i]) ** 2) # Withought sqrt I am comparing distances ^ 2 to distances ^ 2 so closest 3 n withh still be the same
        if pot < cur:
            distances_labels[2][0] = pot
            distances_labels[2][1] = labels_batch[i]
            distances_labels.sort(key=lambda x: x[0])
    return distances_labels

