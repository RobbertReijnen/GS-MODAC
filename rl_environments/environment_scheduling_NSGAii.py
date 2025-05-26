import toml
import json
import pathlib
import os
import pandas as pd

from deap import base, creator, tools
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import GraphInstance

from utils import compute_hypervolume
from scheduling.genetic_algorithm.operators import *

from scheduling.helper_functions import load_job_shop_env
from scheduling.genetic_algorithm.operators import (evaluate_individual, variation,
                                                    init_individual, init_population, mutate_shortest_proc_time,
                                                    mutate_sequence_exchange, pox_crossover)
from rl_environments.helper_functions import *
from settings import SCHEDULING_DATA

REFERENCE_POINTS_FILE = SCHEDULING_DATA + "/reference_points.json"


class schedulingEnv(gym.Env):
    def __init__(self, parameters):
        self.population_size = parameters['environment']['population_size']
        self.nr_objectives = parameters['environment']['nr_objectives']
        self.max_generations = parameters['environment']['max_generations']
        self.alternative_objectives = parameters['environment']['alternative_objectives']
        self.jobShopEnv = None

        self.solution_spaces = []
        self.solution_space = spaces.Box(low=0, high=1, shape=(self.population_size, self.nr_objectives),
                                         dtype=np.float32)
        self.adjacency_matrix = spaces.Box(low=0, high=1, shape=(self.population_size, self.population_size),
                                           dtype=np.int32)

        # Action space: a continuous action for each node with values between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reference_point = None
        self.generation = 0
        self.done = False
        self.reward_factor = parameters['environment']['reward_factor']
        self.problem_instances = parameters['environment']['problem_instances']

        with open(SCHEDULING_DATA + '/ideal_points_{}_obj.json'.format(self.nr_objectives), 'r') as json_file:
            self.ideal_points = json.load(json_file)

        self.save_results = False
        if 'results_saving' in parameters:
            self.save_results = parameters['results_saving']['save_result']
            self.folder = parameters['results_saving']['folder']
            self.exp_name = parameters['results_saving']['exp_name']

        self.bounds = []

    def _observe(self):
        obs = {'graph': GraphInstance(
            nodes=np.array([fit for fit in normalize_fitnesses(self.population, self.bounds)]),
            edges=get_edges(self),
            edge_links=get_edge_links(self)),
            'additional_features': np.array([self.generation / self.max_generations])}
        return obs

    def reset(self):
        self.problem_instance = random.choice(self.problem_instances)
        self.jobShopEnv = load_job_shop_env(self.problem_instance)
        self.toolbox = base.Toolbox()

        creator.create("Fitness", base.Fitness, weights=tuple([-1.0 for i in range(self.nr_objectives)]))
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox.register("init_individual", init_individual, creator.Individual, None, jobShopEnv=self.jobShopEnv)
        self.toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
        self.toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)

        self.toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=self.jobShopEnv)
        self.toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate_individual", evaluate_individual, jobShopEnv=self.jobShopEnv,
                              alt_objectives=self.alternative_objectives, objectives=self.nr_objectives)

        self.population = init_population(self.toolbox, self.population_size, )

        individuals = [[ind[0], ind[1]] for ind in self.population]
        fitnesses = [self.toolbox.evaluate_individual(ind) for ind in individuals]
        fitnesses = [fit[0] for fit in fitnesses]

        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit

        if self.save_results:
            self.hof = tools.ParetoFront()
            self.hof.update(self.population)

        self.bounds = get_fitness_bounds([ind.fitness.values for ind in self.population])
        if not self.save_results:
            self.reference_point = get_fitness_bounds([ind.fitness.values for ind in self.population])[1]
        else:
            if os.path.isfile(REFERENCE_POINTS_FILE):
                with open(REFERENCE_POINTS_FILE, 'r') as file:
                    reference_points = json.load(file)
                    if self.problem_instance in reference_points:
                        if not self.alternative_objectives:
                            self.reference_point = reference_points[self.problem_instance][0:self.nr_objectives]
                        else:
                            print('USING ALTERNATIVE OBJECTIVES (2)')
                            self.reference_point = reference_points[self.problem_instance][-2:]

                        print('using reference point from file', self.reference_point)
                    else:
                        print('NO REFERENCE POINT KNOWN')

        self.generation = 0
        self.done = False

        self.solution_space = np.array([ind.fitness.values for ind in self.population]).astype(np.float32)
        self.solution_spaces.append(self.solution_space)
        self.initial_hv = self.step_hv = compute_hypervolume(self.population, self.nr_objectives, self.reference_point)
        if not self.save_results:
            self.ideal_hv = compute_hypervolume([tuple(self.ideal_points[self.problem_instance][:self.nr_objectives])],
                                                self.nr_objectives, self.reference_point)
        self.best_hv = self.initial_hv

        # Return the initial state
        return self._observe(), dict()

    def step(self, action):
        action1 = np.nan_to_num(action[0])
        action1 = np.clip(action1, -1, 1)
        action2 = np.nan_to_num(action[1])
        action2 = np.clip(action2, -1, 1)

        reward = 0
        self.generation += 1
        cxpb = (action1 + 1) * 0.2 + 0.6  # (between 0.6 and 1)
        mutpb = (action2 + 1) * 0.05  # (between 0 and 0.1)

        offspring = variation(self.population, self.toolbox, self.population_size, cxpb, mutpb)

        if '/dafjs/' or '/yfjs/' in self.jobShopEnv.instance_name:
            offspring = repair_precedence_constraints(self.jobShopEnv, offspring)

        # Evaluate the population
        # sequential evaluation of population
        individuals = [[ind[0], ind[1]] for ind in offspring]
        fitnesses = [self.toolbox.evaluate_individual(ind) for ind in individuals]
        fitnesses = [fit[0] for fit in fitnesses]
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        if self.save_results:
            self.hof.update(offspring)

        self.bounds = update_bounds(self.bounds, fitnesses)

        # Select next generation population
        self.population = self.toolbox.select(self.population + offspring, self.population_size)
        self.solution_space = np.array([list(ind.fitness.values) for ind in self.population]).astype(np.float32)
        self.solution_spaces.append(self.solution_space)

        if not self.save_results:
            episode_hv = compute_hypervolume(self.population, self.nr_objectives, self.reference_point)

            if self.best_hv < episode_hv:
                current_gap = (episode_hv - self.initial_hv) / (self.ideal_hv - self.initial_hv) * 100
                previous_gap = (self.best_hv - self.initial_hv) / (self.ideal_hv - self.initial_hv) * 100
                reward = round(((self.reward_factor * current_gap) ** 2) - ((self.reward_factor * previous_gap) ** 2), 1)
                self.best_hv = episode_hv

        if self.generation == self.max_generations:
            self.done = True

            if self.save_results:
                self.save_result()

        return self._observe(), reward, self.done, None, {}

    # --------------------------------------------------------------------------------------------------------------------
    def sample(self):
        """
        Sample random actions and run the environment
        """
        for episode in range(5):
            start_time = time.time()

            print("start episode: ", episode)
            _, _ = self.reset()
            while True:
                action = env.action_space.sample()
                graph, reward, done, _, _ = env.step(action)
                if done:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"Episode {episode} completed in {duration:.2f} seconds")
                    break

    def save_result(self):
        output_dir = self.folder
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Ensure that exp_name includes a slash (/) if needed
        exp_name = self.exp_name.strip("/")

        results = {}
        results['problem_instance'] = self.problem_instance
        results['hypervolume'] = compute_hypervolume(self.hof, self.nr_objectives, self.reference_point)

        results_csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
        df = pd.DataFrame.from_dict(results, orient='index').T
        pd.DataFrame(df).to_csv(results_csv_path, index=False)

# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    config_filepath = "configs/config_scheduling_5_5.toml"
    with open(config_filepath, 'r') as toml_file:
        parameters = toml.load(toml_file)
    env = schedulingEnv(parameters)
    env.sample()
