import numpy as np
from sklearn.metrics import accuracy_score

# returns cumulative metric score per interval
# |abcd|efg|
#  20%  30%
# num_intervals: number of desired intervals, not size per interval
def interval_result(target, pred, metric_func, num_intervals):
  starting_indices = np.floor(np.linspace(0, target.size-1, num_intervals+1)[:-1]).astype(int)
  interval_size = starting_indices[1] - starting_indices[0]

  metrics_at_intervals = []
  for idx_start in starting_indices:
    metrics_at_intervals.append( metric_func(target[0:idx_start+interval_size], pred[0:idx_start+interval_size]) )

  return np.asarray(metrics_at_intervals)

# target = np.array([1,0,1,1,1])
# pred   = np.array([1,1,0,0,0])

# print(interval_result(target, pred, accuracy_score, 2))
