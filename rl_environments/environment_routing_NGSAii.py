import json
import time
import os
import toml
import pathlib
import gymnasium as gym

from deap import base, creator, tools
from gymnasium.spaces import Box
from gymnasium.spaces import GraphInstance

from utils import compute_hypervolume
from routing.genetic_algorithm.operators import *
from routing.helper_functions import *
from rl_environments.helper_functions import *
from settings import ROUTING_DATA


class routingEnv(gym.Env):
    def __init__(self, parameters):
        self.population_size = parameters['environment']['population_size']
        self.nr_objectives = parameters['environment']['nr_objectives']
        self.max_generations = parameters['environment']['max_generations']

        self.solution_spaces = []
        self.solution_space = Box(low=0, high=1, shape=(self.population_size, self.nr_objectives),
                                  dtype=np.float32)

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reference_point = None
        self.generation = 0
        self.done = False
        self.reward_factor = parameters['environment']['reward_factor']
        self.instance_files = parameters['environment']['instance_file']
        self.problem_instances = parameters['environment']['problem_instances']

        self.ideal_points = {}
        size_to_file = {
            'cvrp_20_': 'ideal_points_20_2_obj.json',
            'cvrp_50_': 'ideal_points_50_2_obj.json',
            'cvrp_100_': 'ideal_points_100_2_obj.json',
            'cvrp_200_': 'ideal_points_200_2_obj.json',
            'cvrp_500_': 'ideal_points_500_2_obj.json'
        }

        for key, filename in size_to_file.items():
            if key in parameters['environment']['instance_file']:
                with open(ROUTING_DATA + filename, 'r') as json_file:
                    self.ideal_points = json.load(json_file)
                break

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
        if type(self.instance_files) == list:
            self.instance_file = random.choice(self.instance_files)
        else:
            self.instance_file = self.instance_files
        print('INSTANCE', self.instance_file)
        nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data = read_input_cvrp(self.instance_file, self.problem_instance)
        self.toolbox = base.Toolbox()

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        if not hasattr(creator, "Fitness"):
            creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox.register('indexes', random.sample, range(1, nb_customers + 1), nb_customers)
        self.toolbox.register('individual', tools.initIterate, creator.Individual, self.toolbox.indexes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", ordered_crossover)
        self.toolbox.register("mutate", mutation_shuffle)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register('evaluate', eval_individual_fitness, truck_capacity=truck_capacity,
                         dist_matrix_data=dist_matrix_data, dist_depot_data=dist_depot_data, demands_data=demands_data)

        self.population = self.toolbox.population(self.population_size)

        fitnesses = [list(self.toolbox.evaluate(i)) for i in self.population]
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = tuple(fit)

        if self.save_results:
            self.hof = tools.ParetoFront()
            self.hof.update(self.population)

        self.bounds = get_fitness_bounds([ind.fitness.values for ind in self.population])
        if not self.save_results:
            self.reference_point = get_fitness_bounds([ind.fitness.values for ind in self.population])[1]
        else:
            size_to_file = {
                'cvrp_20_': "reference_points_20_2_obj.json",
                'cvrp_50_': "reference_points_50_2_obj.json",
                'cvrp_100_': "reference_points_100_2_obj.json",
                'cvrp_200_': "reference_points_200_2_obj.json",
                'cvrp_500_': "reference_points_500_2_obj.json",
            }

            REFERENCE_POINTS_FILE = None
            for key, filename in size_to_file.items():
                if key in self.instance_file:
                    REFERENCE_POINTS_FILE = ROUTING_DATA + filename
                    break

            if os.path.isfile(REFERENCE_POINTS_FILE):
                with open(REFERENCE_POINTS_FILE, 'r') as file:
                    reference_points = json.load(file)
                    if str(self.problem_instance) in reference_points:
                        print('using reference point from file')
                        self.reference_point = reference_points[str(self.problem_instance)]
                        print('ref_point:', self.reference_point)
                    else:
                        print('NO REFERENCE POINT KNOWN')

        self.generation = 0
        self.done = False

        self.solution_space = np.array([ind.fitness.values for ind in self.population]).astype(np.float32)
        self.solution_spaces.append(self.solution_space)
        self.initial_hv = self.step_hv = compute_hypervolume(self.population, self.nr_objectives, self.reference_point)
        if not self.save_results:
            self.ideal_hv = compute_hypervolume([tuple(self.ideal_points[self.instance_file][str(self.problem_instance)])],
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
        cxpb = (action1+1) * 0.1 + 0.5  # (between 0.5 and 0.7)
        mutpb = (action2+1) * 0.1  # (between 0 and 0.2)

        offspring = []
        for _ in range(self.population_size):
            if random.random() <= cxpb:
                ind1, ind2 = list(map(self.toolbox.clone, random.sample(self.population, 2)))
                self.toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values

            else:
                ind1 = self.toolbox.clone(random.choice(self.population))

            self.toolbox.mutate(ind1, mutpb)
            del ind1.fitness.values
            offspring.append(ind1)

        fitnesses = [list(self.toolbox.evaluate(i)) for i in offspring]
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = tuple(fit)

        if self.save_results:
            self.hof.update(offspring)

        self.bounds = update_bounds(self.bounds, fitnesses)

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
                print('saving results')
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
                action = env.action_space.sample()  # Take random action
                graph, reward, done, _, _ = env.step(action)
                if done:
                    end_time = time.time()  # End time for the episode
                    duration = end_time - start_time  # Calculate the duration
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
        print('hv:', results['hypervolume'], 'ref_point:', self.reference_point, len(self.hof))

        results_csv_path = os.path.join(output_dir, f'{exp_name}_results.csv')
        df = pd.DataFrame.from_dict(results, orient='index').T
        pd.DataFrame(df).to_csv(results_csv_path, index=False)


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    config_filepath = "configs/config_routing.toml"
    with open(config_filepath, 'r') as toml_file:
        parameters = toml.load(toml_file)
    env = routingEnv(parameters)
    env.sample()