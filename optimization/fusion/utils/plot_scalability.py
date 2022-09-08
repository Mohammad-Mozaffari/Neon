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


if __name__ == "__main__":
    file_path = os.getcwd() + "/../../../build/"
    file_name = [file for file in os.listdir(file_path) if file.endswith("json")][-1]

    with open(file_path + file_name) as file:
        data = json.load(file)
        plot_complete_figure(data, xlabel="FusionFactors", ylabel="SpeedupsFusionFactor", threshold=10, fit_curve=True,
                             curve_type="Linear")
        plot_complete_figure(data, xlabel="FusionFactors", ylabel="BaselineTimes", threshold=10, fit_curve=False)
        plot_complete_figure(data, xlabel="FusionFactors", ylabel="FusedTime", threshold=10, fit_curve=False)
        plot_complete_figure(data, xlabel="GPUCount", ylabel="SpeedupsGPUCount", fit_curve=False)
        plt.show()

