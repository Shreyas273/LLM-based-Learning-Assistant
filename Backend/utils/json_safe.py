import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]

    return obj