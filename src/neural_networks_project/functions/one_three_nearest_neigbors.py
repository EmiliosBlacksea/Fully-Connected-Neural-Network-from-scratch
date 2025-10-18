import numpy as np
from math import inf
import time


from functions.conv import unpickle
from functions.compare_to_each import compare_to_each_3nn
from functions.compare_distances import distance_sorter
#will access all data compare using 1nn and 3nn and return success rates and time taken
def one_three_nearest_neighbors():
    _start_time = time.time()

    # Load data
    image_data_1 = unpickle('src/neural_networks_project/rawdata/data_batch_1')
    image_data_2 = unpickle('src/neural_networks_project/rawdata/data_batch_2')
    image_data_3 = unpickle('src/neural_networks_project/rawdata/data_batch_3')
    image_data_4 = unpickle('src/neural_networks_project/rawdata/data_batch_4')
    image_data_5 = unpickle('src/neural_networks_project/rawdata/data_batch_5')
    image_labels = unpickle('src/neural_networks_project/rawdata/batches.meta')
    training_data = unpickle('src/neural_networks_project/rawdata/test_batch')
    image_1_batch = image_data_1.get(b'data').astype(np.float32)
    image_2_batch = image_data_2.get(b'data').astype(np.float32)
    image_3_batch = image_data_3.get(b'data').astype(np.float32)
    image_4_batch = image_data_4.get(b'data').astype(np.float32)
    image_5_batch = image_data_5.get(b'data').astype(np.float32)
    training_batch = training_data.get(b'data').astype(np.float32)
    names = image_labels.get(b'label_names')

    # Compare n image from training batch to batches 1-5
    # compare_to_each_3nn function the training batch, their labels,
    #  a list of images to compare it with and their labels
    print("Comparing to batch 1...")
    distances_batch1 = compare_to_each_3nn(training_batch, image_1_batch, image_data_1.get(b'labels'))
    print("Comparing to batch 2...")
    distances_batch2 = compare_to_each_3nn(training_batch, image_2_batch, image_data_2.get(b'labels'))
    print("Comparing to batch 3...")
    distances_batch3 = compare_to_each_3nn(training_batch, image_3_batch, image_data_3.get(b'labels'))
    print("Comparing to batch 4...")
    distances_batch4 = compare_to_each_3nn(training_batch, image_4_batch, image_data_4.get(b'labels'))
    print("Comparing to batch 5...")
    distances_batch5 = compare_to_each_3nn(training_batch, image_5_batch, image_data_5.get(b'labels'))

    #Get 3 nearest neighbors from all batches
    alldistances = [[distances_batch1], [distances_batch2], [distances_batch3], [distances_batch4], [distances_batch5]]

    distances = distance_sorter(alldistances, len(training_batch))


    actual_vs_predicted = []
    for i in range(len(distances)):
        actual_label = training_data.get(b'labels')[i]
        neighbor_labels = [distances[i][3], distances[i][4], distances[i][5]]
        actual_vs_predicted.append([actual_label, neighbor_labels])
    #Get results

    #Get results for 1 nearest neighbor

    success_rate_1nn = 0
    for i in range(len(actual_vs_predicted)):
        decision = True if actual_vs_predicted[i][0] == actual_vs_predicted[i][1][0] else False
        success_rate_1nn += 1 if decision else 0
    
    success_rate_1nn = success_rate_1nn / len(actual_vs_predicted) if success_rate_1nn != 0 else 0
    print(f"1-NN Success Rate: {success_rate_1nn * 100:.2f}%")



    image_1 = image_data_1.get(b'data')[0]
    image_1_label = names[image_data_1.get(b'labels')[0]]

    #Get results for 3 nearest neighbors
    success_rate_3nn = 0
    for i in range(len(actual_vs_predicted)):
        if actual_vs_predicted[i][1][0] != actual_vs_predicted[i][1][1] != actual_vs_predicted[i][1][2]:
            decision = True if actual_vs_predicted[i][0] in actual_vs_predicted[i][1] else False
        if actual_vs_predicted[i][1][0] == actual_vs_predicted[i][1][1] or actual_vs_predicted[i][1][0] == actual_vs_predicted[i][1][2]:
            decision = True if actual_vs_predicted[i][0] == actual_vs_predicted[i][1][0] else False
        if actual_vs_predicted[i][1][1] == actual_vs_predicted[i][1][2]:
            decision = True if actual_vs_predicted[i][0] == actual_vs_predicted[i][1][1] else False
        if actual_vs_predicted[i][1][0] == actual_vs_predicted[i][1][1] == actual_vs_predicted[i][1][2]:
            decision = True if actual_vs_predicted[i][0] == actual_vs_predicted[i][1][0] else False
        success_rate_3nn += 1 if decision else 0
    success_rate_3nn = success_rate_3nn / len(actual_vs_predicted) if success_rate_3nn != 0 else 0
    print(f"3-NN Success Rate: {success_rate_3nn * 100:.2f}%")

    _total_time = time.time() - _start_time
    return success_rate_1nn, success_rate_3nn, _total_time
    

