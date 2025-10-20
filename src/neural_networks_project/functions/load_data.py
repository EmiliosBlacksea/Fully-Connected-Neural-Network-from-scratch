from functions.unpickle import Unpickle
import numpy as np

#RETURNS labels_batch_1->5, data_batch_1->5, labels_training, data_training
def Load_data():
    #Get labels for "training" images
    labels_batch_1 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_1')[b'labels'])
    labels_batch_2 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_2')[b'labels'])
    labels_batch_3 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_3')[b'labels'])
    labels_batch_4 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_4')[b'labels'])
    labels_batch_5 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_5')[b'labels'])
    #Get data for "training" images
    data_batch_1 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_1')[b'data'])
    data_batch_2 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_2')[b'data'])
    data_batch_3 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_3')[b'data'])
    data_batch_4 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_4')[b'data'])
    data_batch_5 = np.array(Unpickle('src/neural_networks_project/rawdata/data_batch_5')[b'data'])
    #Get labels and datafor testing images
    labels_testing = np.array(Unpickle('src/neural_networks_project/rawdata/test_batch')[b'labels'])
    data_testing = np.array(Unpickle('src/neural_networks_project/rawdata/test_batch')[b'data'])
    
    return (
            labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
            data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5, 
            labels_testing, data_testing
            )
       