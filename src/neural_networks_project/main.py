import time
import os
import json

from functions.one_three_nearest_neigbors import one_three_nearest_neighbors
from functions.compare_centers import compare_centers
from functions.compare_centers import nearest_centers_test

#returns success rates and time taken
#success_rate_1nn, success_rate_3nn, _total_time = one_three_nearest_neighbors()
# results = {
#         "1nn": float(success_rate_1nn),
#         "3nn": float(success_rate_3nn),
#         "time_seconds": float(_total_time)
#     }

# _out_path = os.path.join(os.path.dirname(__file__), "run_results.json")
# _tmp = _out_path + ".tmp"
# with open(_tmp, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2)
# os.replace(_tmp, _out_path)

centers, total_train_time = compare_centers()
success_rate_nearest_centers, testing_time = nearest_centers_test(centers)
results = {
        "nearest_centers_success_rate": float(success_rate_nearest_centers),
        "training_time_seconds": float(total_train_time),
        "testing_time_seconds": float(testing_time)
    }
_out_path = os.path.join(os.path.dirname(__file__), "run_results_nearest_centers.json")
_tmp = _out_path + ".tmp"
with open(_tmp, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
os.replace(_tmp, _out_path) 

print(f"Saved results to {_out_path}")