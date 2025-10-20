from functions.load_data import Load_data
from functions.find_3_nn import Find_3_nn
import numpy as np
import time

import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def NN():

    _time = time.time()
    #Getiing all data
    (
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
    ) = Load_data()

    nn3_distances_labels = []    
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
        print(f'Image {i} took {__time:.2f} seconds to calculate')
        if all_nn[1][1] == all_nn[2][1]:
            nnof = all_nn[1][1]
        else:
            nnof = all_nn[0][1]
        log_nn3.append(nnof == labels_train[i])
        log_nn.append(all_nn[0][1] == labels_train[i])
    
    ##
    #Thread attempt
    ##
    # nn3_distances_labels = []
    # log_nn = []
    # log_nn3 = []
    # repetitions = 32

    # start_time = time.time()

    # def compute_one(i):
    #     t0 = time.time()
    #     nn_1 = Find_3_nn(data_train[i], data_batch_1, labels_batch_1)
    #     nn_2 = Find_3_nn(data_train[i], data_batch_2, labels_batch_2)
    #     nn_3 = Find_3_nn(data_train[i], data_batch_3, labels_batch_3)
    #     nn_4 = Find_3_nn(data_train[i], data_batch_4, labels_batch_4)
    #     nn_5 = Find_3_nn(data_train[i], data_batch_5, labels_batch_5)
    #     all_nn = nn_1 + nn_2 + nn_3 + nn_4 + nn_5
    #     all_nn.sort(key=lambda x: x[0])
    #     elapsed = time.time() - t0

    #     # 3-NN vote (your rule) and 1-NN check
    #     nnof = all_nn[1][1] if all_nn[1][1] == all_nn[2][1] else all_nn[0][1]
    #     ok3 = (nnof == labels_train[i])
    #     ok1 = (all_nn[0][1] == labels_train[i])
    #     return i, elapsed, ok3, ok1

    # log_nn3 = [False] * repetitions
    # log_nn = [False] * repetitions

    # with ThreadPoolExecutor(32) as ex: #max_workers=os.cpu_count()
    #     futures = [ex.submit(compute_one, i) for i in range(repetitions)]
    #     for fut in as_completed(futures):
    #         i, elapsed, ok3, ok1 = fut.result()
    #         log_nn3[i] = ok3
    #         log_nn[i] = ok1
    #         print(f"Image {i} took {elapsed:.2f} seconds to calculate")


    # ###
    # ###

    nn3stats = np.mean(log_nn3)
    nnstats = np.mean(log_nn)
    print(time.time() - _time)
    return nn3stats,nnstats, time.time() - _time