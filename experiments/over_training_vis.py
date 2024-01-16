import wandb
from tqdm import tqdm
import torch
from procgen_tools import maze, visualization, models
import os
import shutil
import matplotlib.pyplot as plt

def get_model_number(model_name):
    # model is of format model_<number>:v<version>
    return int(model_name.split('_')[1].split(':')[0])

# Set your specific run ID here
run_id = "jp9tjfzd"
project_name = "procgen"

# Initialize wandb API
api = wandb.Api()

# Fetch the run
run = api.run(f"{project_name}/{run_id}")

# List all artifacts for this run
artifacts = run.logged_artifacts()
for artifact in tqdm(artifacts):
    artifact_to_download = api.artifact(f"{project_name}/{artifact.name}", type="model")
    artifact_dir = artifact_to_download.download()
    model_file = f"{artifact_dir}/{artifact.name[:-3]}.pth"
    model = torch.load(model_file)
    if "state_dict" in model:
        model["model_state_dict"] = model.pop("state_dict")
    torch.save(model, model_file)
    model = models.load_policy(model_file, 15, torch.device("cuda:0"))
    venv = maze.create_venv(1, start_level=0, num_levels=1)
    vf_original = visualization.vector_field(venv, model)
    frames_dir = f"/workspace/maze-values-dev-interp/procgen-tools-dev-interp/experiments/frames"
    os.makedirs(frames_dir, exist_ok=True)
    # Save each plot as an image
    frame_filename = os.path.join(frames_dir, f"frame_{get_model_number(artifact.name)}.png")
    visualization.plot_vf(vf_original)
    plt.savefig(frame_filename)
    plt.close()
    # delete model file
    shutil.rmtree(artifact_dir)