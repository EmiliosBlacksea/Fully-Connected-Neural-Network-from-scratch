import numpy as np
from functions.get_labeled_data import Load_data
from math import inf
import time
(
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
) = Load_data()

def CC():
    _start = time.time()
    
    # Combine all training batches into one array (vectorized)
    all_data = np.vstack([data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5])
    all_labels = np.concatenate([labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5])
    
    # Compute centers for all 10 classes at once (vectorized)
    centers = np.array([all_data[all_labels == c].mean(axis=0) for c in range(10)])
    
    # Compute distances from each test sample to all centers at once (vectorized)
    # Shape: (num_test_samples, 10)
    distances = np.linalg.norm(data_train[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    
    # Find nearest center for each test sample (vectorized)
    predicted_labels = np.argmin(distances, axis=1)
    
    # Compare predictions with actual labels (vectorized)
    log_cc = (predicted_labels == labels_train)
    
    accuracy = np.mean(log_cc)
    total_time = time.time() - _start
    
    print(f"CC Accuracy: {accuracy * 100:.2f}%")
    return accuracy, total_time

