"""
This module start slurm job on the cluster
"""
import os
import json
import argparse
import logging


def transfer_data(direction, which_cluster, folder):
    if which_cluster == "genouest":
        cluster_path = params["genouest_path_to_tfpc"]
    elif which_cluster == "igrida":
        cluster_path = params["igrida_path_to_save_tfpc"]
    elif which_cluster == "nef":
        cluster_path = params["nef_path_to_save_tfpc"]
    else:
        raise RuntimeError("Cluster unknown")
    # We complete the path
    cluster_path += FOLDER_XP + folder + "/"
    local_path = "../" + FOLDER_XP + folder + "/"

    if direction == "download":
        source = cluster_path
        destination = local_path
    elif direction == "upload":
        source = local_path
        destination = cluster_path
    else:
        raise RuntimeError("Only upload and download value correct")

    command = "rsync -r " + source + " " + destination
    print(command)
    os.system(command)


logging.getLogger().setLevel(logging.DEBUG)
FOLDER_XP = "data/models/fine_tune_models/"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--automatic", help="Syncronize all automatically", action="store_true"
)
args = parser.parse_args()

sync_config_file = open("config.json")
params = json.load(sync_config_file)

if not args.automatic:
    list_params_keys = list(params.keys())
    for indice, option in enumerate(list_params_keys):
        print(indice, "-", option)
    indice_from = int(input("Choose from where you want to copy the file : "))
    indice_to = int(input("Choose where do you want to overwrite the folder xp : "))

    FROM_FOLDER_XP = params[list_params_keys[indice_from]] + FOLDER_XP
    TO_FOLDER_XP = params[list_params_keys[indice_to]] + FOLDER_XP

    list_directory = [x[0] for x in os.walk("../" + FOLDER_XP)]
    list_directory = [x.split("/")[-1] for x in list_directory]
    list_directory = [l for l in list_directory if l != ""]
    for indice, direc in enumerate(list_directory):
        print(indice, ":", repr(direc))

    ind_choosen = input("Which directory do you want to copy ?")
    dir_choosen = list_directory[int(ind_choosen)]
    validation = input(
        "You want to replace files in "
        + TO_FOLDER_XP
        + "/ by the files in "
        + FROM_FOLDER_XP
        + dir_choosen
        + " ? (Y to continue)"
    )
    if validation == "Y":
        os.system("scp -r " + FROM_FOLDER_XP + dir_choosen + " " + TO_FOLDER_XP)
    else:
        raise RuntimeError("You cancel the transfer")
elif args.automatic:
    list_directory = [x[0] for x in os.walk("../" + FOLDER_XP)]
    list_directory = [x.split("/")[-1] for x in list_directory]
    list_directory = [l for l in list_directory if l != ""]
    for indice, direc in enumerate(list_directory):
        print(indice, ":", direc)

    # We get all experiences from the different cluster
    logging.info("We get all experiences from the different cluster")
    for directory in list_directory:
        transfer_data("download", "genouest", directory)
        transfer_data("download", "igrida", directory)
        transfer_data("download", "nef", directory)

    # We upload all experiences to all cluster
    logging.info("We upload all experiences to all cluster")
    for directory in list_directory:
        transfer_data("upload", "genouest", directory)
        transfer_data("upload", "igrida", directory)
        transfer_data("upload", "nef", directory)

else:
    raise RuntimeError("Strange to be here")
