import math
import pandas as pd
import numpy as np
from settings import ROUTING_DATA

# load in instance information
def read_input_cvrp(filename, instance_nr):
    print(filename, instance_nr)
    filename = ROUTING_DATA + filename
    data = pd.read_pickle(filename)
    depot_x = data[instance_nr][0][0]
    depot_y = data[instance_nr][0][1]
    customers_x = [x for x, y in data[instance_nr][1]]
    customers_y = [y for x, y in data[instance_nr][1]]
    demands = data[instance_nr][2]
    capacity = data[instance_nr][3]

    distance_matrix = compute_distance_matrix(customers_x, customers_y)
    distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)

    return len(demands), capacity, distance_matrix, distance_depots, demands


# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = np.zeros((nb_customers, nb_customers))
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


# Compute the distances to depot
def compute_distance_depots(depot_x, depot_y, customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        dist = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = dist
    return distance_depots


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return exact_dist