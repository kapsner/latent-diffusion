import os
import pandas as pd
import argparse, logging
from yaml import safe_load, dump, Dumper
from flatten_dict import flatten, unflatten

# https://stackoverflow.com/a/69572347
class dotdict(dict):
  def __setitem__(self, item, value):
    if "." in item:
      head, path = item.split(".", 1)
      obj = self.setdefault(head, dotdict())
      obj[path] = value
    else:
      super().__setitem__(item, value)

def read_grid(csv_file):
    grid = pd.read_csv(
        filepath_or_buffer=os.path.abspath(csv_file),
        dtype=str # import all as strings and define data types when writing
    )
    return grid

def load_config(config_path):
    with open(config_path) as f:
            yaml_content = safe_load(f)
    return yaml_content

def write_config(config_path, content):
    with open(config_path, "w") as f:
        dump(content, f, Dumper=Dumper, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--csv_file",
        dest="csv_file",
        type=str,
        default=None,
        help="The path to the csv file with the settings."
    )
    parser.add_argument(
        "-b",
        "--base",
        dest="base",
        type=str,
        default=None,
        help="The base config file."
    )
    parser.add_argument(
        "-l",
        "--logdir",
        dest="logdir",
        type=str,
        default="/home/user/development/trainings/diffusionmodels",
        help="The logging directory."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # change working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # read grid
    grid = read_grid(args.csv_file)

    # read config yaml
    config_yaml = load_config(args.base)
    config_yaml_flat = flatten(config_yaml)
    
    settings_to_iterate = grid.columns[1:]
    
    # iterate over grid settings
    for _i, _row in grid.iterrows():        
        # updated settings
        for _setting in settings_to_iterate:
            settings_split = _setting.split(".")
            if "." in str(_row[_setting]):
                replace_value = float(_row[_setting])
            else:
                replace_value = int(_row[_setting])
            config_yaml_flat[tuple(settings_split)] = replace_value
        
        # save config yml
        write_config(
            args.base,
            unflatten(config_yaml_flat)
        )

        run_id = "dce_mip-vq-seg" + "_" + _row["param_id"]
        
        # start learning
        cmd_train = "python main.py -b " + args.base + \
            " -t -n " + run_id + \
                " -l " + args.logdir
        os.system(cmd_train)
        
        # identify checkpoint folder
        log_folders = sorted(
            [f for f in os.listdir(args.logdir) if os.path.isdir(
            os.path.join(args.logdir, f)) and run_id in f]
        )
        checkpoint_dir_last_run = os.path.join(args.logdir, log_folders[-1], "checkpoints", "val")
        checkpoints_last_run = sorted(os.listdir(checkpoint_dir_last_run))

        # iterate over best checkpoints and run inference
        for _i in range(0, 2):
            checkpoint_name = checkpoints_last_run[_i]
            output_name = run_id + "_" + checkpoint_name.replace(".ckpt", "")
            full_checkpoint_path = os.path.join(checkpoint_dir_last_run, checkpoint_name)

            cmd_inference = "python sample_diffusion_segmentation.py -c " + full_checkpoint_path + \
                " -b " + args.base + \
                " -n " + output_name + " -o /home/user/development/diffusion_models/sample_images"
            os.system(cmd_inference)
