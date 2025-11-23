from functions.unpickle import Unpickle
import numpy as np

#RETURNS labels_batch_1->5, data_batch_1->5, labels_training, data_training
def Load_data_Cifar10():
    #Get labels for "training" images
   # labels_batch_1 = np.array(Unpickle('..','/rawdata/data_batch_1')[b'labels']).astype(int)
    databatch1 = r"C:\Users\emili\Desktop\NeuralNetworks-Project-1\src\neural_networks_project\rawdata\data_batch_1"
    databatch2 = r"C:\Users\emili\Desktop\NeuralNetworks-Project-1\src\neural_networks_project\rawdata\data_batch_2"
    databatch3 = r"C:\Users\emili\Desktop\NeuralNetworks-Project-1\src\neural_networks_project\rawdata\data_batch_3"
    databatch4 = r"C:\Users\emili\Desktop\NeuralNetworks-Project-1\src\neural_networks_project\rawdata\data_batch_4"
    databatch5 = r"C:\Users\emili\Desktop\NeuralNetworks-Project-1\src\neural_networks_project\rawdata\data_batch_5"
    testbatch = r"C:\Users\emili\Desktop\NeuralNetworks-Project-1\src\neural_networks_project\rawdata\test_batch"
    labels_batch_1 = np.array(Unpickle(databatch1)[b'labels']).astype(int)
    labels_batch_2 = np.array(Unpickle(databatch2)[b'labels']).astype(int)
    labels_batch_3 = np.array(Unpickle(databatch3)[b'labels']).astype(int)
    labels_batch_4 = np.array(Unpickle(databatch4)[b'labels']).astype(int)
    labels_batch_5 = np.array(Unpickle(databatch5)[b'labels']).astype(int)
    #Get data for "training" images
    data_batch_1 = np.array(Unpickle(databatch1)[b'data']).astype(int)
    data_batch_2 = np.array(Unpickle(databatch2)[b'data']).astype(int)
    data_batch_3 = np.array(Unpickle(databatch3)[b'data']).astype(int)
    data_batch_4 = np.array(Unpickle(databatch4)[b'data']).astype(int)
    data_batch_5 = np.array(Unpickle(databatch5)[b'data']).astype(int)
    #Get labels and datafor testing images
    labels_testing = np.array(Unpickle(testbatch)[b'labels']).astype(int)
    data_testing = np.array(Unpickle(testbatch)[b'data']).astype(int)
    
    return (
            labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
            data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5, 
            labels_testing, data_testing
            )
       