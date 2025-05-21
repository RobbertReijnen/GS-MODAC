import random
import pickle


def generate_instance(num_customers, min_demand=1, max_demand=9, capacity=40):
    depot_location = [random.random(), random.random()]
    cust_locations = [[random.random(), random.random()] for cust in range(num_customers)]
    demands = [random.randint(min_demand, max_demand) for cust in range(num_customers)]
    capacity = capacity
    return [depot_location, cust_locations, demands, capacity]


if __name__ == "__main__":
    save = False
    nr_of_instances = 10000
    num_customers = 500

    instances = []

    for instance in range(nr_of_instances):
        instance = generate_instance(num_customers)
        instances.append(instance)

    file = "cvrp_{}_{}.pkl".format(str(num_customers), str(nr_of_instances))

    if save:
        with open(f"{file}", "wb") as file:
            pickle.dump(instances, file)
