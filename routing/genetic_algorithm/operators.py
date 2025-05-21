import random


def get_subroutes(individual, truck_capacity, demands_data):
    """
    Splits a given route into subroutes based on truck capacity and customer demands.

    Parameters:
    individual (list): The list of customer IDs in the route.
    truck_capacity (int): The maximum capacity of the truck.
    demands_data (list): The list of demands for each customer.

    Returns:
    list: A list of subroutes where each subroute is a list of customer IDs.
    """

    routes = []
    sub_route = []
    vehicle_load = 0

    for customer_id in individual:
        demand = demands_data[customer_id - 1]
        if vehicle_load + demand <= truck_capacity:
            sub_route.append(customer_id)
            vehicle_load += demand
        else:
            routes.append(sub_route)
            sub_route = [customer_id]
            vehicle_load = demand

    if sub_route:
        routes.append(sub_route)

    return routes


def compute_routes_fitness(routes, dist_matrix_data, distance_depot_data):
    total_distance = 0
    longest_sub_route = 0
    for route in routes:
        route_distance = 0
        route_distance += distance_depot_data[route[0] - 1] + distance_depot_data[route[-1] - 1]
        for i in range(len(route) - 1):
            route_distance += dist_matrix_data[route[i] - 1][route[i + 1] - 1]
        total_distance += route_distance
        if route_distance > longest_sub_route:
            longest_sub_route = route_distance

    return longest_sub_route, total_distance


def eval_individual_fitness(individual, truck_capacity, dist_matrix_data, dist_depot_data, demands_data):
    routes = get_subroutes(individual, truck_capacity, demands_data)
    longest_sub_route, total_distance = compute_routes_fitness(routes, dist_matrix_data, dist_depot_data)
    return (longest_sub_route, total_distance)


def ordered_crossover(parent1, parent2):
    """
    Performs an ordered crossover between two parent routes in a VRP.

    Parameters:
    parent1 (list): The first parent route.
    parent2 (list): The second parent route.

    Returns:
    tuple: A tuple containing two new child routes.
    """

    # Adjust indices to be 0-based
    ind1 = [x - 1 for x in parent1]
    ind2 = [x - 1 for x in parent2]

    size = min(len(ind1), len(ind2))
    a, b = sorted(random.sample(range(size), 2))

    # Initialize holes
    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # Fill holes
    temp1, temp2 = ind1[:], ind2[:]
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the middle segment
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    ind1 = [x + 1 for x in ind1]
    ind2 = [x + 1 for x in ind2]
    return ind1, ind2


def mutation_shuffle(individual, indpb):
    """
    Mutates an individual by randomly swapping its elements with a given probability.

    Parameters:
    individual (list): The individual to be mutated.
    indpb (float): The probability of each element being swapped.

    Returns:
    tuple: A tuple containing the mutated individual.
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.choice([j for j in range(size) if j != i])
            individual[i], individual[swap_indx] = individual[swap_indx], individual[i]
    return individual,
