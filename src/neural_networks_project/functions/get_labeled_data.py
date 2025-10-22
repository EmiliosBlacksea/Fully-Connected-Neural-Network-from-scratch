import numpy as np
from functions.load_data import Load_data
(
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
    ) = Load_data()

def Get_labeled_data ():
    label0_batch = Get_label(0)
    label1_batch = Get_label(1)
    label2_batch = Get_label(2)
    label3_batch = Get_label(3)
    label4_batch = Get_label(4)
    label5_batch = Get_label(5)
    label6_batch = Get_label(6)
    label7_batch = Get_label(7)
    label8_batch = Get_label(8)
    label9_batch = Get_label(9)
    return

def Get_label(label):
    found = []
    for i in range(data_batch_1):
        if labels_batch_1[i] == label:
            found.append(data_batch_1)
        if labels_batch_2[i] == label:
            found.append(data_batch_1)
        if labels_batch_3[i] == label:
            found.append(data_batch_1)
        if labels_batch_4[i] == label:
            found.append(data_batch_1)
        if labels_batch_5[i] == label:
            found.append(data_batch_1)
    return  found