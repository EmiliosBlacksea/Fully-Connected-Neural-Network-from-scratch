import time
import os
import json

from functions.one_three_nearest_neigbors import one_three_nearest_neighbors

#returns success rates and time taken
success_rate_1nn, success_rate_3nn, _total_time = one_three_nearest_neighbors()

results = {
        "1nn": float(success_rate_1nn),
        "3nn": float(success_rate_3nn),
        "time_seconds": float(_total_time)
    }

_out_path = os.path.join(os.path.dirname(__file__), "run_results.json")
_tmp = _out_path + ".tmp"
with open(_tmp, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
os.replace(_tmp, _out_path)
print(f"Saved results to {_out_path}")