import json
from pathlib import Path


def is_empty_list(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return isinstance(data, list) and len(data) == 0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def remove_file(file_path):
    try:
        file_path.unlink()
        print(f"Removed file: {file_path}")
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")


def check_files(folder):
    for file_path in Path(folder).rglob("results.json"):
        if is_empty_list(file_path):
            remove_file(file_path)


if __name__ == "__main__":
    check_files("experiments/breadth/ATP/")
