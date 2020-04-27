# Computes the interval results for:
# MODEL x FILE SIZE x METRIC

import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score, confusion_matrix

import data_loader
import result_data_loader

# returns cumulative metric score per interval
# |abcd|efg|
#  20%  30%
# target and pred: numpy arrays consisting of 0/1
# num_intervals: number of desired intervals, not size per interval
def interval_result(target, pred, metric_func, num_intervals):
  starting_indices = np.floor(np.linspace(0, target.size-1, num_intervals+1)[:-1]).astype(int)
  interval_size = starting_indices[1] - starting_indices[0]

  metrics_at_intervals = []
  for idx_start in starting_indices:
    metrics_at_intervals.append( metric_func(target[0:idx_start+interval_size], pred[0:idx_start+interval_size]) )

  return np.asarray(metrics_at_intervals)

if __name__ == '__main__':
  program_dir_path = os.path.dirname(os.path.realpath(__file__))
  trace_dir_path   = program_dir_path + "/trace_files/"
  results_dir_path = program_dir_path + "/results/"
  output_dir_path  = results_dir_path + "metrics/"

  model_result_filename_prefix = {"perceptron": "perceptron-",
                                  "slp": "neural-net-single-",
                                  "mlp": "neural-net-ml-",
                                  "lstm": "LSTM-pytorch-",
                                  "gru": "GRU-pytorch-"}

  output_file = open(output_dir_path + "metrics_interval.txt", "w")

  for model in ["perceptron", "slp", "mlp", "lstm", "gru"]:
    for file_size in ["56", "10K"]:
    # for file_size in ["8M"]:
      for metric in [accuracy_score, f1_score, mutual_info_score, confusion_matrix]:
        trace_filename = "gcc-" + file_size + ".txt"
        trace_filepath = trace_dir_path + trace_filename
        results_filepath = results_dir_path + model_result_filename_prefix[model] + trace_filename

        trace_features, trace_labels = data_loader.get_data(data_path=trace_filepath, k=8)
        result_labels = result_data_loader.get_data(data_path=results_filepath)

        metric_output = interval_result(trace_labels, result_labels, metric, 4)

        output_file.write("***> " + model + " " + file_size + " " + metric.__name__ + ":\n")
        output_file.write(str(metric_output))
        output_file.write("\n\n\n")

  output_file.close()
