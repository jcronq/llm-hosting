import uuid
import json

import torch


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def human_readable_memory(memory_bytes: int) -> str:
    memory_bytes = float(memory_bytes)
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    memory_bytes /= 1024
    if memory_bytes < 1024:
        return f"{memory_bytes:.2f} KB"
    memory_bytes /= 1024
    if memory_bytes < 1024:
        return f"{memory_bytes:.2f} MB"
    memory_bytes /= 1024
    if memory_bytes < 1024:
        return f"{memory_bytes:.2f} GB"
    memory_bytes /= 1024
    return f"{memory_bytes:.2f} TB"

def normalize_device_id(device_id):
    if device_id in {"total"}:
        return device_id
    return f"gpu_{device_id}"

def print_cuda_memory():
    device_profile = {"total": {"total": 0, "reserved": 0, "allocated": 0, "free": 0}}
    for i in range(torch.cuda.device_count()):
        memory_profile = {
            "total": torch.cuda.get_device_properties(i).total_memory,
            "reserved": torch.cuda.memory_reserved(i),
            "allocated": torch.cuda.memory_allocated(i),
        }
        memory_profile["free"] = memory_profile["total"] - memory_profile["reserved"] - memory_profile["allocated"]
        for k, v in memory_profile.items():
            device_profile["total"][k] += v
        device_profile[i] = memory_profile
    str_device_profile = {
        normalize_device_id(device_id): {k: human_readable_memory(v) for k, v in memory_profile.items()}
        for device_id, memory_profile in device_profile.items()
    }
    print(json.dumps(str_device_profile, indent=2))
