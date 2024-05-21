# Пункт 3
import multiprocessing
import os
import time
from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import tracemalloc

import worker


def toPDF(results_df):
    pdf = SimpleDocTemplate("results/" + optimizer_name + ".pdf", pagesize=letter)
    table_data = [list(results_df.columns)]
    for i, row in results_df.iterrows():
        table_data.append(list(row))
    table = Table(table_data)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8)])
    table.setStyle(table_style)
    pdf_table = []
    pdf_table.append(table)
    pdf.build(pdf_table)


def wrapper(queue, func, *args, **kwargs):
    start_time = time.time()
    tracemalloc.start()
    res = func(*args, **kwargs)
    end_time = time.time()
    memory_usage = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    elapsed_time = end_time - start_time
    tracemalloc.stop()
    tracemalloc.reset_peak()
    queue.put((res, memory_usage, elapsed_time))


def measure_resources(func, *args, **kwargs):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(queue, func) + args, kwargs=kwargs)
    p.start()
    p.join()
    res, memory_usage, elapsed_time = queue.get()
    return res, memory_usage, elapsed_time


if __name__ == "__main__":
    data = pd.read_csv("housing.csv")
    data.dropna(inplace=True)
    data = data.drop(columns=["ocean_proximity"])
    X = data.drop(['median_house_value'], axis=1)
    Y = data['median_house_value']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)
    optimizers = {"SGD": SGD,}
                  #"Nesterov": SGD,
                  #"Momentum": SGD, }
    # "Adam": Adam,
    # "RMSprop": RMSprop,
    # "Adagrad": Adagrad}

    optimizers_configuration = {"SGD": {"learning_rate": [0.001, 0.0001, 0.00001],
                                        "clipnorm": [1]},
                                "Nesterov": {"learning_rate": [0.001, 0.0001, 0.00001],
                                             "clipnorm": [1],
                                             "nesterov": [True]},
                                "Momentum": {"learning_rate": [0.001, 0.0001, 0.00001],
                                             "clipnorm": [1],
                                             "momentum": [0.1, 0.25, 0.5, 0.75, 0.9]},
                                "Adam": {"learning_rate": [0.001, 0.0001, 0.00001],
                                         "beta_1": [0.9, 0.75, 0.5],
                                         "beta_2": [0.999, 0.9, 0.75],
                                         "amsgrad": [False, True]},
                                "RMSprop": {"learning_rate": [0.001, 0.0001, 0.00001],
                                            "rho": [0.9, 0.75, 0.5],
                                            "centred": [False, True]},
                                "Adagrad": {"learning_rate": [0.001, 0.0001, 0.00001],
                                            "initial_accumulator_value": [0.1, 0.05, 0.25]}}
    # epochs = [8, 64, 512]
    epochs = [2]
    try:
        os.mkdir("results")
    except FileExistsError:
        pass
    overall_results = {}
    for i in epochs:
        overall_results[i] = {}
    for optimizer_name, optimizer_raw in optimizers.items():
        results = []
        for epoch in epochs:
            keys, values = zip(*optimizers_configuration[optimizer_name].items())
            conf_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
            min_loss = float('inf')
            min_loss_config = {}
            for configuration in conf_combinations:
                print(f"Optimizer: {optimizer_name}, Epochs: {epoch}, Configuration:{configuration}")
                optimizer = optimizer_raw(**configuration)
                args = (optimizer, epoch, X_train, X_test, Y_train, Y_test)
                kwargs = {}
                result, memory_usage, elapsed_time = measure_resources(worker.work, *args, **kwargs)
                if result < min_loss:
                    min_loss = result
                    min_loss_config = configuration
                results.append({
                    "Configuration": '\n'.join([f"{key}: {value}" for key, value in configuration.items()]),
                    "Epochs": epoch,
                    "Loss": result,
                    "Memory Usage (MB)": memory_usage,
                    "Elapsed Time (s)": round(elapsed_time, 5)
                })
            overall_results[epoch][optimizer_name] = (min_loss, min_loss_config)
        df = pd.DataFrame(results)
        toPDF(df)

    print("Minimum Loss Value by Epochs amount: ")
    for epochs, data in overall_results.items():
        epo_min_keys = sorted(data, key=lambda k: data[k][0])[:3]
        print(epo_min_keys)
        print(f" Epochs: {epochs}")
        for key in epo_min_keys:
            print(f"  Optimizer: {key}, Loss: {data[key][0]}, Configuration:{data[key][1]}")
