import pandas as pd
import wandb
from tqdm import tqdm

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("luchayward/point-transformer")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

for run in tqdm(runs):
    # Could use pattern match in newer python here
    if "hand_selected_reversed" in run.name:
        run.config["dataset"] = "25%"
        run.update()
    if "5%area" == run.name:
        run.config["dataset"] = "5%"
        run.update()
    if "2.5%area" in run.name:
        run.config["dataset"] = "2.5%"
        run.update()

print("Done")
