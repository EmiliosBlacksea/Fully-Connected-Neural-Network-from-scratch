from NN import NN
from CC import CC
import numpy as np
from functions.neural_network.neural import FC_layer
from functions.load_data import Load_data
from scipy.io import savemat
import time


def horizontal_flip(data_batch):
    imgs = data_batch.reshape(-1, 3, 32, 32)
    flipped = imgs[:, :, :, ::-1]
    return flipped.reshape(-1, 3072)

def cross_entropy_with_grad(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    N, C = y_pred.shape

    if y_true.ndim == 1:
        # integer labels
        loss = -np.log(y_pred[np.arange(N), y_true]).mean()
        grad_z = y_pred.copy()
        grad_z[np.arange(N), y_true] -= 1.0   # y_prob - one_hot
    else:
        # one-hot labels
        loss = -(y_true * np.log(y_pred)).sum() / N
        grad_z = (y_pred - y_true)

    return loss, grad_z

Loss = []
training_accuracy = []
testing_accuracy = []

def testmain():
    (
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
    ) = Load_data()
    epoch = 3
    seasons = 50
    LR = 10**(-4)
    for e in range(epoch):
        e = 2

        WL1 = None
        WL2 = None
        WL3 = None
        WL4 = None
        WL5 = None
        WL6 = None
        bL1 = None
        bL2 = None
        bL3 = None
        bL4 = None
        bL5 = None
        bL6 = None

        if e != 0:
            time.sleep(5)
            WL1 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer1_season{e * seasons + 10}.npy")
            WL2 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer2_season{e * seasons + 10}.npy")
            WL3 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer3_season{e * seasons + 10}.npy")
            WL4 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer4_season{e * seasons + 10}.npy")
            #WL5 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer5_season{e * 10}.npy")
            #WL6 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer6_season{e * 10}.npy")
            bL1 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer1_season{e * seasons + 10}.npy")
            bL2 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer2_season{e * seasons + 10}.npy")
            bL3 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer3_season{e * seasons + 10}.npy")
            bL4 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer4_season{e * seasons + 10}.npy")
            #bL5 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer5_season{e * 10}.npy")
            #bL6 = np.load(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer6_season{e * 10}.npy")

        FC1 = None
        FC2 = None
        FC3 = None
        FC4 = None
        FC5 = None
        FC6 = None
        FC7 = None

        FC1 = FC_layer(Input=3072, Self=1024, batch=50, LR=LR, activation="relu", W=WL1, b=bL1)
        FC2 = FC_layer(Input=1024, Self=512, batch=50,LR=LR, activation="relu", W=WL2, b=bL2)
        FC3 = FC_layer(Input=512, Self=256, batch=50,LR=LR, activation="relu", W=WL3, b=bL3)
        FC4 = FC_layer(Input=256, Self=10, batch=50,LR=LR, activation="softmax", W=WL4, b=bL4)
        # FC5 = FC_layer(Input=100, Self=100, batch=50,LR=LR, activation="relu", W=WL5, b=bL5)
        # FC6 = FC_layer(Input=100, Self=10, batch=50,LR=LR, activation="softmax", W=WL6, b=bL6)

        for i in range(seasons):
            (
                data_batch, labels_batch
            ) = shuffle()
            print("season", e * seasons + i + 1)
            for j in range(len(data_batch)):
    
                print("Training batch", j + 1, ".", end=" ")
                train(data_batch[j],labels_batch[j], FC1=FC1, FC2=FC2, FC3=FC3, FC4=FC4,FC5=FC5, FC6=FC6, FC7=FC7, dropout = True, keepProp=0.75)
                print("Training ", end=" ")
                test(data_batch[j],labels_batch[j], FC1=FC1, FC2=FC2, FC3=FC3, FC4=FC4,FC5=FC5, FC6=FC6, FC7=FC7, training=True,)
                print("Testing ", end=" ")
                test(data_train,labels_train, FC1=FC1, FC2=FC2, FC3=FC3, FC4=FC4,FC5=FC5, FC6=FC6, FC7=FC7, training=False)
                print()

            if (e * seasons + i + 1) % 10 == 0:   
                layers = [FC1, FC2, FC3, FC4, FC5, FC6, FC7]
                active = [L for L in layers if L is not None]   
                k = 1
                for L in (active):
                    save_W_b(L, k, e * seasons + i + 1)
                    k += 1

        if e == 1:
            LR = 3*10**(-4)
        if e == 2:
            LR = 10**(-4)
        
        savemat("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/testing_accuracy.mat", {"data": testing_accuracy})
        savemat("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/training_accuracy.mat", {"data": training_accuracy})
        savemat("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Loss.mat", {"data": Loss})
        break

    return FC1, FC2, FC3

def train(data_batch,labels_batch, FC1 = None, FC2 = None , FC3 = None, FC4 = None, FC5 = None, FC6 = None, FC7 = None, dropout = False, keepProp = 1):
    batch = 50
    i = 0
    Los = []
    layers = [FC1, FC2, FC3, FC4, FC5, FC6, FC7]
    active = [L for L in layers if L is not None]
    tot_layers = len(active)
    while i < len(data_batch): #Generate input and target
        if dropout:
            masks = []
        target = np.zeros((batch, 10), np.float32)
        for j in range(batch):
            target[j][labels_batch[i + j]] = 1.0
        inp = data_batch[i: i + batch]

        i += batch
        #print(inp.dtype, inp.min(), inp.max())#####
        inp = inp.astype(np.float32) / 255.0
        y = inp
        j = 0
        for L in (active): #Forward Pass
            y = L.forward(y)
            if (j < tot_layers - 1) and dropout:
                rand_mat = np.random.rand(*y.shape)   # ίδιο shape με a2
                masks.append(rand_mat < keepProp)
                y = y * masks[j] / keepProp 
                j += 1

        loss, dE_dz = cross_entropy_with_grad(y, target)
         
        grad = dE_dz
        if dropout:
            mask_idx = len(masks) - 1
        for L in reversed(active):#Bacckward Pass
            grad = L.backward(grad)
            if dropout and mask_idx >= 0:
                grad = grad * masks[mask_idx] / keepProp
                mask_idx -= 1

           
        Los.append(loss)
        #print(loss)
        if i % 10000 == 0 or i == 5 :
            Loss.append(np.mean(Los))
            print(f"Loss = { np.mean(Los):.2f}.", end=" ")

def test(data_batch, labels_batch, FC1 = None, FC2 = None , FC3 = None, FC4 = None, FC5 = None, FC6 = None, FC7 = None,  training = True):#, FC2, FC3, FC4, FC5
    total = len(data_batch)
    correct = 0
    pr = []
    layers = [FC1, FC2, FC3, FC4, FC5, FC6, FC7]
    active = [L for L in layers if L is not None]
    for i in range(total):
        target = np.zeros((1, 10), np.float32)
        target[0,labels_batch[i]] = 1.0

        y = np.array([data_batch[i]]).astype(np.float32)
        
        for L in (active):#forwatd Pass
            y = L.forward(y)

        pred = int(np.argmax(y, axis=1)[0])
        true = int(np.argmax(target, axis=1)[0])
        pr.append(pred)
        if pred == true:
            correct += 1
    if training == True:
        training_accuracy.append(correct / total * 100)
    if training == False:
        testing_accuracy.append(correct / total * 100)
    print(f"accuracy: = {correct/total * 100:.2f}%.", end=" ")

def shuffle():
    (
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
    ) = Load_data()
    an, bn = merge_shuffle_split([data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5, horizontal_flip(data_batch_1), horizontal_flip(data_batch_2), horizontal_flip(data_batch_3), horizontal_flip(data_batch_4), horizontal_flip(data_batch_5)], [labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5, labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5])
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = an
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = bn
    return [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10], [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]

def merge_shuffle_split(data_batches, label_batches, *, seed=None):
    """
    Take a list of batches, merge them, shuffle globally, then split back
    into batches with the SAME SIZES as the inputs.
    """
    if not (isinstance(data_batches, (list, tuple)) and isinstance(label_batches, (list, tuple))):
        raise ValueError("Pass data_batches and label_batches as lists/tuples of arrays")

    if len(data_batches) != len(label_batches):
        raise ValueError("data_batches and label_batches must have same length")

    # original batch sizes (e.g. [10_000, 10_000, 10_000, ...])
    sizes = [int(b.shape[0]) for b in data_batches]

    # 1) fuse all batches
    concat_data = np.concatenate(data_batches, axis=0)   # shape (N, ...)
    concat_labels = np.concatenate(label_batches, axis=0)  # shape (N, ...)

    total = concat_data.shape[0]
    if concat_labels.shape[0] != total:
        raise ValueError(f"Total samples mismatch: data has {total}, labels has {concat_labels.shape[0]}")

    if sum(sizes) != total:
        raise ValueError(f"Sum of batch sizes {sum(sizes)} != total samples {total}. "
                         "Check your input batch shapes")

    # 2) ONE global permutation over ALL samples
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)

    shuffled_data = concat_data[perm]
    shuffled_labels = concat_labels[perm]

    # 3) split back according to original sizes
    out_data = []
    out_labels = []
    idx = 0
    for s in sizes:
        out_data.append(shuffled_data[idx: idx + s])
        out_labels.append(shuffled_labels[idx: idx + s])
        idx += s

    return out_data, out_labels

def save_W_b(FC, n, s):
    W, b = FC.get_W_b()
    np.save(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer{n}_season{s}.npy", W)
    np.save(f"C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer{n}_season{s}.npy", b)

def testsave():
    (
    labels_batch_1, labels_batch_2, labels_batch_3, labels_batch_4, labels_batch_5,
    data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5,
    labels_train,
    data_train
    ) = Load_data()
    WL1 = np.load("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer1.npy")
    WL2 = np.load("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/Weights_Layer2.npy")
    bL1 = np.load("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer1.npy")
    bL2 = np.load("C:/Users/emili/Desktop/NeuralNetworks-Project-1/results/bias_Layer2.npy")
    LR = 10**(-5)
    FC1 = FC_layer(3072, 100, 40, 2, LR, "relu", WL1, bL1)
    FC2 = FC_layer(100, 10, 40, 2, LR, "softmax", WL2, bL2)
    print("Testing ", end=" ")
    test(data_train, labels_train, FC1, FC2)
    print()

testmain()