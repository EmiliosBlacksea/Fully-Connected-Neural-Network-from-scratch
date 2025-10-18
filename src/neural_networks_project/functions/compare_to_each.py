from functions.distance import euclidean_distance
from math import inf

#GETS the training batch, their labels, a list of images to compare it with and their labels

def compare_to_each_3nn(training_batch, image_batches, image_labels_batches):
    distances = []
    for i in range(len(training_batch)):
        best3distances = [inf, inf, inf, 0, 0, 0] # label
        for j in range(len(image_batches)):
            dist = euclidean_distance(training_batch[i], image_batches[j])
            if dist < best3distances[2]:
                best3distances[2] = dist
                best3distances[5] = image_labels_batches[j]
                # Sort distances
                for k in range(2, 0, -1):
                    if best3distances[k] < best3distances[k - 1]:
                        # Swap distances and labels
                        best3distances[k], best3distances[k - 1] = best3distances[k - 1], best3distances[k]
                        best3distances[k + 3], best3distances[k + 2] = best3distances[k + 2], best3distances[k + 3]
                    else:
                        break
        distances.append(best3distances)
    return distances