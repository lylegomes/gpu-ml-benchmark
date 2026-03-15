import pynvml

def get_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    return {
        "gpu": utilization.gpu,
        "memory": utilization.memory
    }