# Store all the paths for the project
import os.path

# All paths are relative to the root of the repository
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATA_GENERATION_CFGS_PATH = os.path.join(REPO_PATH, "data_generation_cfgs")

DATASET_BASE_DIR = os.path.join(REPO_PATH, "data_trajectories")

SCRIPTS_PATH = os.path.join(REPO_PATH, "scripts")
