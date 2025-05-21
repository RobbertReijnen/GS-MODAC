import random
import os


def generate_fjsp_instance(num_jobs, num_machines, min_num_operations_per_job, max_num_operations_per_job,
                           min_processing_time, max_processing_time):
    instance = []
    machines = [str(i) for i in range(1, num_machines + 1)]

    # Append the header line with the number of jobs and machines as strings
    instance.append(f"{num_jobs} {num_machines}")

    for _ in range(num_jobs):
        job_operations = []
        num_operations_per_job = random.randint(min_num_operations_per_job, max_num_operations_per_job)

        for _ in range(num_operations_per_job):
            num_machine_options = random.randint(1, num_machines)
            machine_options = random.sample(machines, num_machine_options)
            operation_line = f"{num_machine_options} "

            for machine_id in machine_options:
                duration = random.randint(min_processing_time, max_processing_time)
                operation_line += f"{machine_id} {duration} "

            job_operations.append(operation_line.strip())

        job_line = f"{num_operations_per_job} {' '.join(job_operations)}"
        instance.append(job_line)

    return "\n".join(instance)


if __name__ == "__main__":
    save = True
    instances_type = 'test'
    nr_of_instances = 100
    num_jobs = 5
    num_machines = 5
    min_num_operations_per_job = 4
    max_num_operations_per_job = 8
    min_processing_time = 2
    max_processing_time = 20

    for id in range(nr_of_instances):
        folder = "{}/j{}_m{}".format(instances_type, num_jobs, num_machines)
        file = "{}_j{}_m{}_{}".format(instances_type, num_jobs, num_machines, id)

        os.makedirs(folder, exist_ok=True)
        fjsp_instance = generate_fjsp_instance(num_jobs, num_machines, min_num_operations_per_job,
                                               max_num_operations_per_job, min_processing_time, max_processing_time)

        if save:
            with open(f"{folder}/{file}.txt", "w") as instance_file:
                instance_file.write(fjsp_instance)
