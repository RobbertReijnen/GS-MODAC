# ADAPTED FROM: https://github.com/krishna-praveen/Capacitated-Vehicle-Routing-Problem/tree/master

import argparse
import json
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd

from deap import base, creator, tools
from routing.helper_functions import read_input_cvrp
from routing.genetic_algorithm.operators import eval_individual_fitness, ordered_crossover, mutation_shuffle
from scheduling.helper_functions import record_stats
from utils import compute_hypervolume

from settings import ROUTING_DATA

logging.basicConfig(level=logging.INFO)

PARAM_FILE = "configs/NSGAii_routing.json"
DEFAULT_RESULTS_ROOT = "./results/routing_runs"


def save_results(hof, logbook, folder, exp_name, kwargs):
    output_dir = folder
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    exp_name = exp_name.strip("/")
    logbook_csv_path = os.path.join(output_dir, f'{exp_name}_logbook.csv')
    logbook_df = pd.DataFrame(logbook)
    logbook_df.to_csv(logbook_csv_path, index=False)

    hof_data = []
    for ind in hof:
        hof_data.append(ind.fitness.values)

    hof_df = pd.DataFrame(hof_data, columns=[f'Objective_{i + 1}' for i in range(len(hof_data[0]))])
    hof_csv_path = os.path.join(output_dir, f'{exp_name}_hof.csv')
    hof_df.to_csv(hof_csv_path, index=False)

    for i in range(len(hof[0].fitness.values)):
        kwargs[f'min_obj_{i}'] = min([ind.fitness.values[i] for ind in hof])
        kwargs[f'max_obj_{i}'] = max([ind.fitness.values[i] for ind in hof])

    results_csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
    df = pd.DataFrame.from_dict(kwargs, orient='index').T
    pd.DataFrame(df).to_csv(results_csv_path, index=False)


def initialize_run(**kwargs):
    nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data = read_input_cvrp(kwargs['instance_file'], kwargs['problem_instance'])

    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox.register('indexes', random.sample, range(1, nb_customers + 1), nb_customers)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", ordered_crossover)
    toolbox.register("mutate", mutation_shuffle)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register('evaluate', eval_individual_fitness, truck_capacity=truck_capacity, dist_matrix_data=dist_matrix_data, dist_depot_data=dist_depot_data, demands_data=demands_data)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    initial_population = toolbox.population(n=kwargs['population_size'])
    fitnesses = list(map(toolbox.evaluate, initial_population))

    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    return initial_population, toolbox, stats, hof


def run_algo(population, toolbox, folder, exp_name, stats, hof, **kwargs):

    hof.update(population)

    gen = 0
    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # Update the statistics with the new population
    record_stats(gen, population, logbook, stats, kwargs['logbook'], df_list, logging)

    for gen in range(1, kwargs['ngen']+1):

        offspring = []
        for _ in range(kwargs['population_size']):
            if random.random() <= kwargs['cr']:
                ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values

            else:
                ind1 = toolbox.clone(random.choice(population))

            toolbox.mutate(ind1, kwargs['indpb'])
            del ind1.fitness.values
            offspring.append(ind1)

        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)
        population[:] = toolbox.select(population + offspring, kwargs['population_size'])
        record_stats(gen, population, logbook, stats, kwargs['logbook'], df_list, logging)

    # Load existing reference point and compute hypervolume
    size_to_file = {
        'cvrp_20_': "reference_points_20_2_obj.json",
        'cvrp_50_': "reference_points_50_2_obj.json",
        'cvrp_100_': "reference_points_100_2_obj.json",
        'cvrp_200_': "reference_points_200_2_obj.json",
        'cvrp_500_': "reference_points_500_2_obj.json",
    }

    reference_points = None
    for key, filename in size_to_file.items():
        if key in kwargs['instance_file']:
            reference_points = ROUTING_DATA + filename
            break

    if os.path.isfile(reference_points):
        with open(reference_points, 'r') as file:
            reference_points = json.load(file)
            if str(kwargs['problem_instance']) in reference_points:
                reference_point = reference_points[str(kwargs['problem_instance'])]
                hypervolume = compute_hypervolume(hof, kwargs['nr_of_objectives'], list(reference_point))
                kwargs['hypervolume'] = hypervolume
            else:
                print('NO REFERENCE POINT KNOWN')

    if folder is not None:
        save_results(hof, logbook, folder, exp_name, kwargs)

    return hypervolume


def main(param_file=PARAM_FILE):
    try:
        parameters = json.load(open(param_file))
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    folder = (
            DEFAULT_RESULTS_ROOT
            + "/"
            + str(parameters['problem_instance'])
            + "/ngen"
            + str(parameters["ngen"])
            + "_pop"
            + str(parameters['population_size'])
            + "_cr"
            + str(parameters["cr"])
            + "_indpb"
            + str(parameters["indpb"])
    )

    exp_name = "rseed" + str(parameters['rseed'])
    population, toolbox, stats, hof = initialize_run(**parameters)
    run_algo(population, toolbox, folder, exp_name, stats, hof, **parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for Routing")
    parser.add_argument(
        "--config_file",
        metavar='f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="Path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)

