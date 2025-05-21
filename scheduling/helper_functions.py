import os
import json

import pandas as pd
import pathlib
from scheduling.scheduling_environment.jobShop import JobShop
from scheduling.data_parsers import parser_fjsp, parser_fajsp_sdsts


def load_parameters(config_json):
    """Load parameters from a json file"""
    with open(config_json, "rb") as f:
        config_params = json.load(f)
    return config_params


def load_job_shop_env(problem_instance: str, from_absolute_path=False) -> JobShop:

    jobShopEnv = JobShop()
    if 'test' in problem_instance or 'train' in problem_instance:
        print(problem_instance)
        jobShopEnv = parser_fjsp.parse(jobShopEnv, problem_instance, from_absolute_path=False)
    elif '/fajsp_sdsts/' in problem_instance:
        jobShopEnv = parser_fajsp_sdsts.parse(jobShopEnv, problem_instance, from_absolute_path=False)
    else:
        raise NotImplementedError(
            f"""Problem instance {
            problem_instance
            } not implemented"""
        )
    jobShopEnv._name = problem_instance
    return jobShopEnv


def create_stats_list(population, gen):
    stats_list = []
    for ind in population:
        tmp_dict = {}
        tmp_dict.update(
            {
                "Generation": gen,
                "obj1": ind.fitness.values[0]
            })
        if hasattr(ind, "objectives"):
            tmp_dict.update(
                {
                    "obj1": ind.objectives[0],
                }
            )
        tmp_dict = {**tmp_dict}
        stats_list.append(tmp_dict)
    return stats_list


def record_stats(gen, population, logbook, stats, verbose, df_list, logging):
    stats_list = create_stats_list(population, gen)
    df_list.append(pd.DataFrame(stats_list))
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, **record)
    if verbose:
        logging.info(logbook.stream)


def update_operations_available_for_scheduling(env):
    scheduled_operations = set(env.scheduled_operations)
    precedence_relations = env.precedence_relations_operations
    operations_available = [
        operation
        for operation in env.operations
        if operation not in scheduled_operations and all(
            prec_operation in scheduled_operations
            for prec_operation in precedence_relations[operation.operation_id]
        )
    ]
    env.set_operations_available_for_scheduling(operations_available)


def dict_to_excel(dictionary, folder, filename):
    """Save outputs in files"""

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Check if the file exists, if so, give a warning
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        print(f"Warning: {full_path} already exists. Overwriting file.")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([dictionary])

    # Save the DataFrame to Excel
    df.to_excel(full_path, index=False, engine='openpyxl')


def save_results(hof, logbook, folder, exp_name, params):
    """Save outputs in files"""
    output_dir = folder + exp_name
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # add best solution objectives to the parameters
    for i in range(len(hof[0].fitness.values)):
        params['min_obj_{}'.format(i)] = [min([ind.fitness.values[i] for ind in hof])]
        params['max_obj_{}'.format(i)] = [max([ind.fitness.values[i] for ind in hof])]

    # if len(hof[0].fitness.values) == 3:
    #     params['hypervolume'] = [hypervolume([ind.fitness.values for ind in hof.items], tuple(params['reference_point']))]
    # else:
    #     print('cannot compute hv with other nr objective than 3')

    # params['reference_point'] = [params['reference_point']]

    # params['pareto_front'] = [list(set([ind.fitness.values for ind in hof.items]))]

    print('debug')
    df = pd.DataFrame.from_dict(params, columns=None)
    pd.DataFrame(df).to_csv(output_dir + '/result.csv')

    # DEAP Logbook
    pd.DataFrame(logbook).to_csv(output_dir + "/logbook.csv", index=False)

def select_first_operation_for_scheduling(operation):
    return operation.operation_options[0]

