from functions.dumby import dummy_function 
from functions.conv import unpickle
from functions.distance import euclidean_distance
import numpy as np
from math import inf
import time
import json
import os


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

# Compare n image from training batch to all images
distances = []
print(len(training_batch))
for i in range(len(training_batch)):
    best3distances = [inf, inf, inf, 0, 0, 0] # label
    print(i)

    for j in range(len(image_1_batch)):
        dist = euclidean_distance(training_batch[i], image_1_batch[j])
        if dist < best3distances[2]:
            best3distances[2] = dist
            best3distances[5] = image_data_1.get(b'labels')[j]
            # Sort distances
            for k in range(2, 0, -1):
                if best3distances[k] < best3distances[k - 1]:
                    # Swap distances and labels
                    best3distances[k], best3distances[k - 1] = best3distances[k - 1], best3distances[k]
                    best3distances[k + 3], best3distances[k + 2] = best3distances[k + 2], best3distances[k + 3]
                else:
                    break
    for j in range(len(image_2_batch)):
        dist = euclidean_distance(training_batch[i], image_2_batch[j])
        if dist < best3distances[2]:
            best3distances[2] = dist
            best3distances[5] = image_data_2.get(b'labels')[j]
            # Sort distances
            for k in range(2, 0, -1):
                if best3distances[k] < best3distances[k - 1]:
                    # Swap distances and labels
                    best3distances[k], best3distances[k - 1] = best3distances[k - 1], best3distances[k]
                    best3distances[k + 3], best3distances[k + 2] = best3distances[k + 2], best3distances[k + 3]
                else:
                    break
    for j in range(len(image_3_batch)):
        dist = euclidean_distance(training_batch[i], image_3_batch[j])
        if dist < best3distances[2]:
            best3distances[2] = dist
            best3distances[5] = image_data_3.get(b'labels')[j]
            # Sort distances
            for k in range(2, 0, -1):
                if best3distances[k] < best3distances[k - 1]:
                    # Swap distances and labels
                    best3distances[k], best3distances[k - 1] = best3distances[k - 1], best3distances[k]
                    best3distances[k + 3], best3distances[k + 2] = best3distances[k + 2], best3distances[k + 3]
                else:
                    break
    for j in range(len(image_4_batch)):
        dist = euclidean_distance(training_batch[i], image_4_batch[j])
        if dist < best3distances[2]:
            best3distances[2] = dist
            best3distances[5] = image_data_4.get(b'labels')[j]
            # Sort distances
            for k in range(2, 0, -1):
                if best3distances[k] < best3distances[k - 1]:
                    # Swap distances and labels
                    best3distances[k], best3distances[k - 1] = best3distances[k - 1], best3distances[k]
                    best3distances[k + 3], best3distances[k + 2] = best3distances[k + 2], best3distances[k + 3]
                else:
                    break
    for j in range(len(image_5_batch)):
        dist = euclidean_distance(training_batch[i], image_5_batch[j])
        if dist < best3distances[2]:
            best3distances[2] = dist
            best3distances[5] = image_data_5.get(b'labels')[j]
            # Sort distances
            for k in range(2, 0, -1):
                if best3distances[k] < best3distances[k - 1]:
                    # Swap distances and labels
                    best3distances[k], best3distances[k - 1] = best3distances[k - 1], best3distances[k]
                    best3distances[k + 3], best3distances[k + 2] = best3distances[k + 2], best3distances[k + 3]
                else:
                    break
    
    distances.append(best3distances)

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

results = {
    "1nn": float(success_rate_1nn),
    "3nn": float(success_rate_3nn),
    "time_seconds": float(_total_time)
}

_out_path = os.path.join(os.path.dirname(__file__), "run_results.json")
_tmp = _out_path + ".tmp"
with open(_tmp, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
os.replace(_tmp, _out_path)

print(f"Saved results to {_out_path}")

