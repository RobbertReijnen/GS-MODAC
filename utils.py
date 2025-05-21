import torch
import toml
import os
import datetime
import shutil
import socket
import random

import pandas as pd
import numpy as np

from pymoo.indicators.hv import Hypervolume


def load_config(config_filepath):
    with open(config_filepath, 'r') as toml_file:
        return toml.load(toml_file)


def setup_device(config):
    device_type = config['general']['device']
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_type)
    print(f"Device set to: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    return device


def create_log_directory(config, config_file):
    model_logdir = os.path.join("rl_environments/trained_models",
                                config['ppo']['training_comment'] + "/" +
                                datetime.datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname()
                                + "_r" + str(random.randint(0,100000))
                                )
    os.makedirs(model_logdir, exist_ok=True)
    os.makedirs(model_logdir + '/intermediate_models/', exist_ok=True)
    os.makedirs(model_logdir + '/best_model/', exist_ok=True)
    shutil.copyfile(config_file, os.path.join(model_logdir, "train_config.toml"))
    print('started training with directory:', model_logdir)
    return model_logdir


def compute_hypervolume(population, nr_objectives, reference_point):
    """
    :param population:
    :return: compute the hypervolume of a population
    """
    if hasattr(population[0], 'fitness'):
        data = []
        objectives = [f'OBJ{i + 1}' for i in range(nr_objectives)]
        for individual in population:
            data_dict = {obj: individual.fitness.values[count] for count, obj in enumerate(objectives)}
            data.append(data_dict)

        df = pd.DataFrame(data)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Create the pointset
        pointset = np.array([[df[obj].iloc[i] for obj in objectives] for i in range(len(df))])

    else:
        # used in case of an ideal point (which is not a population, but just a point).
        pointset = np.array(population)

    # Compute hypervolume
    ref_point = np.array(reference_point)
    hv = Hypervolume(ref_point=ref_point)
    hypervolume = hv.do(pointset)
    return round(hypervolume, 5)


def repair(ind):
    for i in range(len(ind)):
        if ind[i] < 0.0:
            ind[i] = 0.0
        elif ind[i] > 1.0:
            ind[i] = 1.0
    return ind
