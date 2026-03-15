import torch
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
from models import load_model
from metrics import get_gpu_utilization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)


models_to_test = ["resnet50", "mobilenet_v2", "efficientnet_b0"]
for model_name in models_to_test:

    model = load_model(model_name).to(device)

    filename = f"{model_name}_results.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["batch_size", "latency", "throughput"])
        batch_sizes = [1,8,16,32,64]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

            # warmup
            for _ in range(10):
                model(input_tensor)

            torch.cuda.synchronize()
            start = time.time()

            runs = 100
            for _ in range(runs):
                model(input_tensor)

            torch.cuda.synchronize()
            end = time.time()

            latency = (end - start) / runs
            throughput = batch_size / latency

            gpu_stats = get_gpu_utilization()

            print(f"Batch size: {batch_size}")
            print(f"Latency: {latency:.4f}")
            print(f"Throughput: {throughput:.2f}")
            print("GPU Utilization:", gpu_stats)
            print("-----")
            writer.writerow([batch_size, f"{latency:.4f}", f"{throughput:.2f}"])

    data = pd.read_csv(filename)

    plt.plot(data["batch_size"], data["throughput"])
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (images/sec)")
    plt.title("GPU Inference Throughput Scaling")

    plot_filename = f"{model_name}_throughput_scaling.png"
    plt.savefig(plot_filename)
    plt.clf()