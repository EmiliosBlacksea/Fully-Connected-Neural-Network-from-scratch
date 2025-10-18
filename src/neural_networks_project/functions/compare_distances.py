from math import inf 
#gets an 3dimensional array taht has all batches, all distances
#and an number for images in training batch
#returns sorted distances with their labels
def distance_sorter(alldistances, rep):
    results = []
    for i in range(rep):
        best3distances = [inf, inf, inf, 0, 0, 0]  # label
        for batch in alldistances:
            dist = batch[0][i]
            for k in range(3):
                if dist[k] < best3distances[2]:
                    best3distances[2] = dist[k]
                    best3distances[5] = dist[k + 3]
                    # Sort distances
                    for m in range(2, 0, -1):
                        if best3distances[m] < best3distances[m - 1]:
                            # Swap distances and labels
                            best3distances[m], best3distances[m - 1] = best3distances[m - 1], best3distances[m]
                            best3distances[m + 3], best3distances[m + 2] = best3distances[m + 2], best3distances[m + 3]
                        else:
                            break
        results.append(best3distances)
    return results