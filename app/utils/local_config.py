import json
import os

# from dotenv import load_dotenv

# load env variables
# load_dotenv()

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

config_filepath = os.path.join(base_path, "local_config.json")
global LOCAL_CONFIG
LOCAL_CONFIG = json.load(open(config_filepath))
# global PROJECT_NAME
# PROJECT_NAME = os.environ.get("WANDB_PROJECT")


def write_local_config(PROJECT_NAME, art_type, art_name, art_folder, art_ext):
    # write name
    LOCAL_CONFIG[PROJECT_NAME] = {
        "artifacts": {f"{art_type}": {"filename": f"{art_name}"}}
    }
    # write folder
    root_dir = os.path.join(base_path, "artifacts", art_folder)
    LOCAL_CONFIG[PROJECT_NAME]["artifacts"][f"{art_type}"]["root_dir"] = root_dir
    # write filepath
    filepath = os.path.join(root_dir, art_name + art_ext)
    LOCAL_CONFIG[PROJECT_NAME]["artifacts"][f"{art_type}"]["filepath"] = filepath

    # save updates to CONFIG file
    with open(config_filepath, "w") as f:
        json.dump(LOCAL_CONFIG, f, indent=2)

    return filepath


# def load_local_config():


#     return
