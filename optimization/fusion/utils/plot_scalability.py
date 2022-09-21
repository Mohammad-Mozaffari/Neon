import os
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def linear_objective(x, a, b):
    return a * x + b


def reciprocal_objective(x, a, b):
    return a / x + b


def find_index(fusion_factors, target_val):
    for i in range(len(fusion_factors)):
        if fusion_factors[i] >= target_val:
            return i
    return len(fusion_factors)


def plot_scalability(x, y, threshold, xlabel="", ylabel="", fit_curve=True, new_fig=False, curve_type="Linear"):
    if new_fig:
        plt.figure()
    plt.scatter(x, y)
    if fit_curve:
        objective = linear_objective if curve_type == "Linear" else reciprocal_objective
        if threshold:
            knee_point = find_index(x, threshold)
            popt, _ = curve_fit(linear_objective, x[:knee_point], y[:knee_point])
            a, b = popt
            popt, _ = curve_fit(linear_objective, x[knee_point:], y[knee_point:])
            c, d = popt
        else:
            popt, _ = curve_fit(linear_objective, x, y)
            a, b = popt
            c, d = popt
        x = np.linspace(0.1, max(x), 100)
        plt.plot(x, np.minimum(objective(x, a, b), objective(x, c, d)))
        # plt.legend(["Samples Points", "Fitted Curve"])
        # plt.title(f"Speedup = {a} FusionFactor + {b}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_complete_figure(data, xlabel, ylabel, threshold=0, fit_curve=False, curve_type="Linear"):
    plt.figure()
    grid_size = 32
    while f"{xlabel}_{grid_size}" in data:
        plot_scalability(data[f"{xlabel}_{grid_size}"], data[f"{ylabel}_{grid_size}"], threshold, xlabel=xlabel, ylabel=ylabel,
                         fit_curve=fit_curve, curve_type=curve_type)
        grid_size *= 2
    legends = []
    while grid_size > 32:
        grid_size /= 2
        if fit_curve:
            legends = [f"Samples Points - Grid Size: {grid_size}", f"Fitted Curve - Grid Size: {grid_size}"] + legends
        else:
            legends = [f"Samples Points - Grid Size: {grid_size}"] + legends
    plt.legend(legends)


def generate_vector(data, target_label, target_vals, default_vals, value_name="fused_time"):
    value_idx = {"speedup": 0, "baseline_time": 1, "fused_time": 2}[value_name]
    target_string = ""
    for label, val in default_vals:
        if label == target_label:
            target_string += target_label + "{}_"
        else:
            target_string += label + str(val) + "_"
    target_string = target_string[:-1]

    result = []
    
    for target_val in target_vals:
        result.append(data[target_string.format(target_val)][value_idx])
    return result




if __name__ == "__main__":
    file_path = os.getcwd() + "/../../../build/"
    file_name = [file for file in os.listdir(file_path) if file.endswith("json")][-1]

    with open(file_path + file_name) as file:
        data = json.load(file)
        # plot_complete_figure(data, xlabel="FusionFactors", ylabel="SpeedupsFusionFactor", threshold=10, fit_curve=True,
        #                      curve_type="Linear")
        # plot_complete_figure(data, xlabel="FusionFactors", ylabel="BaselineTimes", threshold=10, fit_curve=False)
        # plot_complete_figure(data, xlabel="FusionFactors", ylabel="FusedTime", threshold=10, fit_curve=False)
        # plot_complete_figure(data, xlabel="GPUCount", ylabel="SpeedupsGPUCount", fit_curve=False)
        gpu_counts = [1, 2, 3, 4]
        memory_access_cnts = [1]
        flop_cnts = [1, 2, 4, 8]
        fusion_factors = [1, 2, 4, 8, 16, 32, 64]
        domain_sizes = [32, 64, 128, 256]
        default_vals = [("GPUCount:", 1), ("MemoryAccessCount:", 1), ("FlopCount:", 1), ("FusionFactor:", 32), ("DomainSize:", 128)]
        
        plt.figure()
        legends = []
        default_indices = [0, 1, 2]
        for domain_size in domain_sizes:
            default_vals[4] = ("DomainSize:", domain_size)
            baseline_time = np.array(generate_vector(data, "FusionFactor:", [1], default_vals, "baseline_time"))
            fused_times = np.array(generate_vector(data, "FusionFactor:", fusion_factors, default_vals))
            speedups = baseline_time / fused_times
            plt.plot(fusion_factors, speedups)
            plt.xlabel("Fusion Factor")
            plt.ylabel("Speedup")
            plt.title([default_vals[i] for i in default_indices])
            legends.append(f"Domain Size: {domain_size}")
        plt.legend(legends)


        plt.figure()
        legends = []
        default_indices = [0, 1, 2]
        baseline_time = np.array(generate_vector(data, "FusionFactor:", [1], default_vals, "baseline_time"))
        for domain_size in domain_sizes:
            default_vals[4] = ("DomainSize:", domain_size)
            fused_times = np.array(generate_vector(data, "FusionFactor:", fusion_factors, default_vals))
            plt.plot(fusion_factors, fused_times)
            plt.xlabel("Fusion Factor")
            plt.ylabel("Time (ms)")
            plt.title([default_vals[i] for i in default_indices])
            legends.append(f"Domain Size: {domain_size}")
        plt.legend(legends)

        plt.figure()
        default_vals[4] = ("DomainSize:", 256)
        default_indices = [1, 2, 4]
        legends = []
        baseline_time = np.array(generate_vector(data, "GPUCount:", [1], default_vals, "baseline_time"))
        print(baseline_time)
        for fusion_factor in fusion_factors:
            default_vals[3] = ("FusionFactor:", fusion_factor)
            fused_times = np.array(generate_vector(data, "GPUCount:", gpu_counts, default_vals))
            speedups = baseline_time / fused_times
            plt.plot(gpu_counts, speedups)
            plt.xlabel("#GPUs")
            plt.ylabel("Speedup")
            plt.title([default_vals[i] for i in default_indices])
            legends.append(f"Fusion Factor: {fusion_factor}")
        plt.legend(legends)


        plt.figure()
        default_vals[0] = ("GPUCount:", 1)
        default_vals[3] = ("FusionFactor:", 32)
        legends = []
        print(baseline_time)
        default_indices = [0, 1, 3] 
        for domain_size in domain_sizes:
            default_vals[4] = ("DomainSize:", domain_size)
            fused_times = np.array(generate_vector(data, "FlopCount:", flop_cnts, default_vals))
            baseline_time = np.array(generate_vector(data, "FlopCount:", [1], default_vals, "baseline_time"))
            # speedups = baseline_time / fused_times
            speedups = np.array(generate_vector(data, "FlopCount:", flop_cnts, default_vals, "speedup"))
            plt.plot(flop_cnts, speedups)
            plt.xlabel("Arithmetic Intensity")
            plt.ylabel("Speedup")
            plt.title([default_vals[i] for i in default_indices])
            legends.append(f"Domain Size: {domain_size}")
        plt.legend(legends)

        plt.show()

