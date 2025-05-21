import os
import re
import random
import csv


def generate_sdsts(path, folder, file, save):
    with open(path + '/' + folder + '/' + file, "r") as data:
        sdsts = []
        total_operations, _, total_machines = re.findall('\S+', data.readline())
        for machine in range(int(total_machines)):
            setuptime_machine = [[random.randint(5,25) for _ in range(int(total_operations))] for _ in range(int(total_operations))]
            sdsts.append(setuptime_machine)

    if save:
        with open(path + "/sdsts/" + file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(sdsts)


if __name__ == "__main__":
    save = True
    folder = 'dafjs'  # Replace with your folder path

    current_directory = os.getcwd()
    file_list = os.listdir(folder)  # Get a list of all files and directories in the folder

    for file in file_list:
        generate_sdsts(current_directory, folder, file, save)
