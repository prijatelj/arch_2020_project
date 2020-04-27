# Computes the overall results for:
# MODEL x FILE SIZE x METRIC
import os
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score, confusion_matrix, matthews_corrcoef
import data_loader
import result_data_loader
from tqdm import tqdm


def overall_result(target, pred, metric_func):
    return metric_func(target, pred)


if __name__ == '__main__':
    program_dir_path = os.path.dirname(os.path.realpath(__file__))
    trace_dir_path = program_dir_path + "/trace_files/"
    results_dir_path = program_dir_path + "/results/"
    output_dir_path = results_dir_path + "metrics/"

    model_result_filename_prefix = {"perceptron": "perceptron-",
                                    "nn": "neural-net-single-",
                                    "ml-nn": "neural-net-ml-",
                                    "lstm": "LSTM-pytorch-",
                                    "gru": "GRU-pytorch-"}

    output_file = open(output_dir_path + "metrics_overall.txt", "w")

    for model in tqdm(["perceptron", "nn", "ml-nn", "lstm", "gru"]):
        for file_size in ["56", "10K", "8M"]:
            trace_filename = "gcc-" + file_size + ".txt"
            trace_filepath = trace_dir_path + trace_filename
            results_filepath = results_dir_path + model_result_filename_prefix[model] + trace_filename

            trace_features, trace_labels = data_loader.get_data(data_path=trace_filepath, k=8)
            result_labels = result_data_loader.get_data(data_path=results_filepath)

            for metric in [accuracy_score, f1_score, mutual_info_score, confusion_matrix, matthews_corrcoef]:
                metric_output = overall_result(trace_labels, result_labels, metric)

                output_file.write("***> " + model + " " + file_size + " " + metric.__name__ + ":\n")
                output_file.write(str(metric_output))
                output_file.write("\n\n\n")

    output_file.close()

    # (m,n) baseline
    output_file_mn = open(output_dir_path + "metrics_interval_mn.txt", "w")

    for file_size in ["56", "10K", "8M"]:
        trace_filename = "gcc-" + file_size + ".txt"
        trace_filepath = trace_dir_path + trace_filename

        results_filepath = ""
        if file_size == "56":
            results_filepath = results_dir_path + "python-gcc-56-(3,1).dat"
        elif file_size == "10K":
            results_filepath = results_dir_path + "python-gcc-10K-(4,1).dat"
        else:
            results_filepath = results_dir_path + "python-gcc-8M-(6,2).dat"

        trace_features, trace_labels = data_loader.get_data(data_path=trace_filepath, k=8)
        result_labels = result_data_loader.get_data(data_path=results_filepath)

        for metric in [accuracy_score, f1_score, mutual_info_score, confusion_matrix, matthews_corrcoef]:
            metric_output = overall_result(trace_labels, result_labels, metric)

            output_file_mn.write("***> " + file_size + " " + metric.__name__ + ":\n")
            output_file_mn.write(str(metric_output))
            output_file_mn.write("\n\n\n")

    output_file_mn.close()
