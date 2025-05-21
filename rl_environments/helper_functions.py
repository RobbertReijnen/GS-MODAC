import numpy as np
from deap import tools


def get_fitness_bounds(fitnesses):
    fitnesses = np.array(fitnesses)

    # Calculate min and max bounds
    min_bounds = np.min(fitnesses, axis=0)
    max_bounds = np.max(fitnesses, axis=0)

    # Calculate bounds
    bounds = np.array([min_bounds, max_bounds])
    return bounds


def update_bounds(bounds, fitness):
    fitness = np.array(fitness)

    # Calculate min and max bounds for each objective separately
    min_bounds = np.min(fitness, axis=0)
    max_bounds = np.max(fitness, axis=0)

    # Update bounds for each objective separately
    for i in range(len(min_bounds)):
        bounds[0][i] = np.minimum(bounds[0][i], min_bounds[i])
        bounds[1][i] = np.maximum(bounds[1][i], max_bounds[i])

    return bounds


def get_edge_links(self):
    edge_links = [[], []]
    for i in range(self.population_size):
        for j in range(self.population_size):
            # if i != j:
            edge_links[0].append(i)
            edge_links[1].append(j)
    return np.array(edge_links).astype(np.int64)


def get_edges(self):
    # Perform non-dominated sorting on the population
    fronts = tools.sortNondominated(self.population, self.population_size, first_front_only=False)

    # Create a mapping from individual to its front
    individual_to_front = {i: [] for i in range(len(fronts))}
    for front_index, front in enumerate(fronts):
        for individual in front:
            individual_index = self.population.index(individual)
            individual_to_front[front_index].append(individual_index)

    # Initialize the adjacency matrix
    edges = []
    for i in range(self.population_size):
        for j in range(self.population_size):
            edges.append(False)

    # Connect individuals within the same Pareto front
    for front, individuals in individual_to_front.items():
        for i in individuals:
            for j in individuals:
                if i != j:
                    edges[i * self.population_size + j] = True
    return np.array(edges).astype(np.bool_)


def normalize_fitnesses(population, bounds):
    normalized_fitnesses = []
    for ind in population:
        normalized_values = []
        for obj, fit in enumerate(ind.fitness.values):
            if bounds[0][obj] > bounds[1][obj]:
                raise ValueError('bound min > bound max', bounds[0][obj], bounds[1][obj], bounds)
            elif bounds[0][obj] == bounds[1][obj]:
                normalized_values.append(0.5)
            else:
                normalized_values.append((fit - bounds[0][obj]) / (bounds[1][obj] - bounds[0][obj]))
        normalized_fitnesses.append(normalized_values)
    return normalized_fitnesses
