import uuid


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def human_readable_memory(memory_bytes: int) -> str:
    memory_bytes = float(memory_bytes)
    if memory_bytes < 1024:
        return f'{memory_bytes} B'
    memory_bytes /= 1024
    if memory_bytes < 1024:
        return f'{memory_bytes:.2f} KiB'
    memory_bytes /= 1024
    if memory_bytes < 1024:
        return f'{memory_bytes:.2f} MiB'
    memory_bytes /= 1024
    if memory_bytes < 1024:
        return f'{memory_bytes:.2f} GiB'
    if memory_bytes < 1024:
        return f'{memory_bytes:.2f} TiB'
    memory_bytes /= 1024
    return f'{memory_bytes:.2f} PiB'

def normalize_device_id(device_id):
    if isinstance(device_id, int):
        return f"gpu_{device_id}"
    return device_id

def sprint_byte_object(obj, offset = 0, immediate_offset = 0):
    print_str = ""
    if isinstance(obj, dict):
        print_str += "{\n"
        sub_str = ""
        for k, v in obj.items():
            sub_str += " " * (offset+2) + f'"{normalize_device_id(k)}": '
            sub_str += sprint_byte_object(v, offset + 2, 0)
        print_str += sub_str[:-2] + "\n"
        if offset == 0:
            print_str += "}\n"
        else:
            print_str += ((" " *offset) + "},\n")
    elif isinstance(obj, list):
        print_str += "[\n"
        sub_str = ""
        for v in obj:
            sub_str += sprint_byte_object(v, offset + 4, offset + 4)
        print_str += sub_str[:-2] + "\n"
        print_str += (" " *offset) + "],\n"
    elif isinstance(obj, str):
        print_str += (" " * immediate_offset) + f'"{obj}",\n'
    elif isinstance(obj, (int, float)):
        print_str += sprint_byte_object(human_readable_memory(obj), offset, immediate_offset)
    elif isinstance(obj, (tuple | set)):
        print_str += sprint_byte_object(list(obj), offset, immediate_offset)
    return print_str

if __name__ == "__main__":
    print(sprint_byte_object({"total": {"free": 100_000_000, "foo": [1_024, 1_000, 1_000_000], "bar": {"one", "two"}}, "fizz": ("buzz",)}))