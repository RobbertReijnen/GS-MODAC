import os
import json
import argparse
import logging
import pathlib
import multiprocessing

import pandas as pd
import numpy as np

from multiprocessing.pool import Pool
from deap import base, creator, tools

from utils import compute_hypervolume

from scheduling.helper_functions import record_stats, load_parameters, load_job_shop_env
from scheduling.genetic_algorithm.operators import (evaluate_population, evaluate_individual, variation,
                                                   init_individual, init_population, mutate_shortest_proc_time,
                                                   mutate_sequence_exchange, pox_crossover, repair_precedence_constraints)

logging.basicConfig(level=logging.INFO)

PARAM_FILE = "configs/NSGAii_scheduling.json"
DEFAULT_RESULTS_ROOT = "./results/single_runs"
REFERENCE_POINTS_FILE = os.getcwd() + "/data/reference_points.json"


def save_results(hof, logbook, folder, exp_name, kwargs):
    output_dir = folder
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    exp_name = exp_name.strip("/")
    logbook_csv_path = os.path.join(output_dir, f'{exp_name}_logbook.csv')
    logbook_df = pd.DataFrame(logbook)
    logbook_df.to_csv(logbook_csv_path, index=False)

    # Create a DataFrame for the hall of fame data
    hof_data = []
    for ind in hof:
        hof_data.append(ind.fitness.values)

    hof_df = pd.DataFrame(hof_data, columns=[f'Objective_{i + 1}' for i in range(len(hof_data[0]))])
    hof_csv_path = os.path.join(output_dir, f'{exp_name}_hof.csv')
    hof_df.to_csv(hof_csv_path, index=False)

    # add best solution objectives to the parameters
    for i in range(len(hof[0].fitness.values)):
        kwargs['min_obj_{}'.format(i)] = min([ind.fitness.values[i] for ind in hof])
        kwargs['max_obj_{}'.format(i)] = max([ind.fitness.values[i] for ind in hof])

    results_csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
    df = pd.DataFrame.from_dict(kwargs, orient='index').T
    pd.DataFrame(df).to_csv(results_csv_path,index=False)


def initialize_run(pool: Pool, **kwargs):
    """Initializes the run by setting up the environment, toolbox, statistics, hall of fame, and initial population.

    Args:
        pool: Multiprocessing pool.
        kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the initial population, toolbox, statistics, hall of fame, and environment.
    """
    try:
        jobShopEnv = load_job_shop_env(kwargs['problem_instance'])
    except FileNotFoundError:
        logging.error(f"Problem instance {kwargs['problem_instance']} not found.")
        return

    toolbox = base.Toolbox()
    if pool != None:
        toolbox.register("map", pool.map)

    creator.create("Fitness", base.Fitness, weights=tuple([-1.0 for i in range(kwargs['nr_of_objectives'])]))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox.register("init_individual", init_individual, creator.Individual, kwargs, jobShopEnv=jobShopEnv)
    toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
    toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
    toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)

    toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=jobShopEnv)
    toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate_individual", evaluate_individual, jobShopEnv=jobShopEnv, objectives=kwargs['nr_of_objectives'], alt_objectives=kwargs['alternative_objectives'])

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    initial_population = init_population(toolbox, kwargs['population_size'], )
    try:
        fitnesses = evaluate_population(toolbox, initial_population, kwargs['nr_of_objectives'], logging)
    except Exception as e:
        logging.error(f"An error occurred during initial population evaluation: {e}")
        return

    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    return initial_population, toolbox, stats, hof, jobShopEnv


def run_algo(jobShopEnv, population, toolbox, folder, exp_name, stats=None, hof=None, **kwargs):
    """Executes the genetic algorithm and returns the best individual.

    Args:
        jobShopEnv: The problem environment.
        population: The initial population.
        toolbox: DEAP toolbox.
        folder: The folder to save results in.
        exp_name: The experiment name.
        stats: DEAP statistics (optional).
        hof: Hall of Fame (optional).
        kwargs: Additional keyword arguments.

    Returns:
        The best individual found by the genetic algorithm.
    """

    hof.update(population)

    gen = 0
    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # Update the statistics with the new population
    record_stats(gen, population, logbook, stats, kwargs['logbook'], df_list, logging)

    if kwargs['logbook']:
        logging.info(logbook.stream)

    for gen in range(1, kwargs['ngen'] + 1):
        # Vary the population
        offspring = variation(population, toolbox, kwargs['population_size'], kwargs['cr'], kwargs['indpb'])

        # Ensure that precedence constraints between jobs are satisfied (only for assembly scheduling (fajsp))
        if '/dafjs/' or '/yfjs/' in jobShopEnv.instance_name:
            offspring = repair_precedence_constraints(jobShopEnv, offspring)

        # Evaluate the population
        fitnesses = evaluate_population(toolbox, offspring, kwargs['nr_of_objectives'], logging)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select next generation population
        population[:] = toolbox.select(population + offspring, kwargs['population_size'])
        # Update the statistics with the new population
        record_stats(gen, population, logbook, stats, kwargs['logbook'], df_list, logging)

    # Load existing reference point and compute hypervolume
    if os.path.isfile(REFERENCE_POINTS_FILE):
        with open(REFERENCE_POINTS_FILE, 'r') as file:
            reference_points = json.load(file)
            if kwargs['problem_instance'] in reference_points:
                if not kwargs['alternative_objectives']:
                    reference_point = reference_points[kwargs['problem_instance']][0:kwargs['nr_of_objectives']]
                else:
                    print('USING ALTERNATIVE OBJECTIVES (2)')
                    reference_point = reference_points[kwargs['problem_instance']][-2:]
                hypervolume = compute_hypervolume(hof, kwargs['nr_of_objectives'], list(reference_point))
                kwargs['hypervolume'] = hypervolume
            else:
                print('NO REFERENCE POINT KNOWN')

    if folder != None:
        save_results(hof, logbook, folder, exp_name, kwargs)

    return hypervolume


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return
    
    pool = multiprocessing.Pool()
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

    exp_name = ("rseed" + str(parameters["rseed"]))
    population, toolbox, stats, hof, jobShopEnv = initialize_run(pool, **parameters)
    run_algo(jobShopEnv, population, toolbox, folder, exp_name, stats, hof, **parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
