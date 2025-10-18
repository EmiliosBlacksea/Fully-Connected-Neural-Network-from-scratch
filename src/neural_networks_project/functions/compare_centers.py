import numpy as np
from math import inf
import time


from functions.conv import unpickle


def compare_centers():
    start_time = time.time()
    # Load data
    image_data_1 = unpickle('src/neural_networks_project/rawdata/data_batch_1')
    image_data_2 = unpickle('src/neural_networks_project/rawdata/data_batch_2')
    image_data_3 = unpickle('src/neural_networks_project/rawdata/data_batch_3')
    image_data_4 = unpickle('src/neural_networks_project/rawdata/data_batch_4')
    image_data_5 = unpickle('src/neural_networks_project/rawdata/data_batch_5')
    
    # make batch of each label's images
    label0_batch = label_finder(0, image_data_1, [])
    label0_batch = label_finder(0, image_data_2, label0_batch)
    label0_batch = label_finder(0, image_data_3, label0_batch)
    label0_batch = label_finder(0, image_data_4, label0_batch)
    label0_batch = label_finder(0, image_data_5, label0_batch)
    label1_batch = label_finder(1, image_data_1, [])
    label1_batch = label_finder(1, image_data_2, label1_batch)
    label1_batch = label_finder(1, image_data_3, label1_batch)
    label1_batch = label_finder(1, image_data_4, label1_batch)
    label1_batch = label_finder(1, image_data_5, label1_batch)
    label2_batch = label_finder(2, image_data_1, [])
    label2_batch = label_finder(2, image_data_2, label2_batch)
    label2_batch = label_finder(2, image_data_3, label2_batch)
    label2_batch = label_finder(2, image_data_4, label2_batch)
    label2_batch = label_finder(2, image_data_5, label2_batch)
    label3_batch = label_finder(3, image_data_1, [])    
    label3_batch = label_finder(3, image_data_2, label3_batch)
    label3_batch = label_finder(3, image_data_3, label3_batch)
    label3_batch = label_finder(3, image_data_4, label3_batch)
    label3_batch = label_finder(3, image_data_5, label3_batch)
    label4_batch = label_finder(4, image_data_1, [])
    label4_batch = label_finder(4, image_data_2, label4_batch)
    label4_batch = label_finder(4, image_data_3, label4_batch)
    label4_batch = label_finder(4, image_data_4, label4_batch)
    label4_batch = label_finder(4, image_data_5, label4_batch)
    label5_batch = label_finder(5, image_data_1, [])
    label5_batch = label_finder(5, image_data_2, label5_batch)
    label5_batch = label_finder(5, image_data_3, label5_batch)
    label5_batch = label_finder(5, image_data_4, label5_batch)
    label5_batch = label_finder(5, image_data_5, label5_batch)
    label6_batch = label_finder(6, image_data_1, [])
    label6_batch = label_finder(6, image_data_2, label6_batch)
    label6_batch = label_finder(6, image_data_3, label6_batch)
    label6_batch = label_finder(6, image_data_4, label6_batch)
    label6_batch = label_finder(6, image_data_5, label6_batch)
    label7_batch = label_finder(7, image_data_1, [])
    label7_batch = label_finder(7, image_data_2, label7_batch)
    label7_batch = label_finder(7, image_data_3, label7_batch)
    label7_batch = label_finder(7, image_data_4, label7_batch)
    label7_batch = label_finder(7, image_data_5, label7_batch)
    label8_batch = label_finder(8, image_data_1, [])
    label8_batch = label_finder(8, image_data_2, label8_batch)
    label8_batch = label_finder(8, image_data_3, label8_batch)
    label8_batch = label_finder(8, image_data_4, label8_batch)
    label8_batch = label_finder(8, image_data_5, label8_batch)
    label9_batch = label_finder(9, image_data_1, [])
    label9_batch = label_finder(9, image_data_2, label9_batch)
    label9_batch = label_finder(9, image_data_3, label9_batch)
    label9_batch = label_finder(9, image_data_4, label9_batch)
    label9_batch = label_finder(9, image_data_5, label9_batch) 

    # calculate center vectors
    
    label0_center = np.mean(label0_batch, axis=0)
    label1_center = np.mean(label1_batch, axis=0)
    label2_center = np.mean(label2_batch, axis=0)
    label3_center = np.mean(label3_batch, axis=0)
    label4_center = np.mean(label4_batch, axis=0)
    label5_center = np.mean(label5_batch, axis=0)
    label6_center = np.mean(label6_batch, axis=0)
    label7_center = np.mean(label7_batch, axis=0)
    label8_center = np.mean(label8_batch, axis=0)
    label9_center = np.mean(label9_batch, axis=0)
    end_time = time.time()
    total_time = end_time - start_time
    centers = [label0_center, label1_center, label2_center, label3_center, label4_center,
               label5_center, label6_center, label7_center, label8_center, label9_center]
    return centers, total_time

def label_finder(label_number, image_data, current_batch):
    image_data_labels = image_data.get(b'labels')
    image_batch = image_data.get(b'data').astype(np.float32)
    for i in range(len(image_data_labels)):
        if image_data_labels[i] == label_number:
            current_batch.append(image_batch[i])
    return current_batch

def nearest_centers_test(centers):
    
    success_rate = 0
    start_time = time.time()
    image_test_data = unpickle('src/neural_networks_project/rawdata/test_batch')
    test_images = image_test_data.get(b'data').astype(np.float32)
    test_labels = image_test_data.get(b'labels')
    predictionvs_actual = []
    success_rate = 0
    for i in range(len(test_images)):
        min_distance = inf
        edicted_label = -1
        for label in range(10):
            center_vector = centers[label]
            distance = np.linalg.norm(test_images[i] - center_vector)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label
        predictionvs_actual.append((predicted_label, test_labels[i]))
        if predicted_label == test_labels[i]:
            success_rate += 1
    success_rate = success_rate / len(test_images)

    end_time = time.time()
    return success_rate, end_time - start_time