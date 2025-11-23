from NN import NN
from CC import CC
import os
import json

ccstats, total_time = CC()

results_cc = {
               "CC-Accuracy": ccstats * 100,
               "Total Time (seconds)": total_time
}

with open('results_cc.json', 'w') as json_file1:
    json.dump(results_cc, json_file1, indent = 4)

nn3stats, nnstats, total_time, log_nn, log_3nn = NN()
print(nn3stats, nnstats, total_time)
results = {
            "3-NN Accuracy": nn3stats * 100,
            "1-NN Accuracy": nnstats * 100,
            "Total Time (seconds)": total_time,
        }
with open('results_int.json', 'w') as json_file2:
    json.dump(results, json_file2, indent=4)
    print("Results saved to results.json")

