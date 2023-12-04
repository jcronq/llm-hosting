import uuid


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
