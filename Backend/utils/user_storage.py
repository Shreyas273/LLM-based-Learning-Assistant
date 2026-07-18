import os
import json

BASE_PATH = "data/users"

def get_user_path(user_id):
    path = f"{BASE_PATH}/{user_id}"
    os.makedirs(path, exist_ok=True)
    return path

def load_user_file(user_id, filename):
    path = f"{get_user_path(user_id)}/{filename}"

    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        return json.load(f)

def save_user_file(user_id, filename, data):
    path = f"{get_user_path(user_id)}/{filename}"

    with open(path, "w") as f:
        json.dump(data, f, indent=2)