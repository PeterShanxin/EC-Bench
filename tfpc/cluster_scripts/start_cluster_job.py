"""
This module start slurm job on the cluster
"""
import os
import argparse

FOLDER_CONFIG_FILE = "data/models/fine_tune_models/"

# Specification of the argument to specify when you launch the script
parser = argparse.ArgumentParser()
parser.add_argument(
    "cluster_type", help="It is a Slurm(slurm) or a Oar cluster(oar) or oar_nef"
)
parser.add_argument("experience_name", help="What is the name of the experience")
parser.add_argument("ram_qt", help="How many ram you want in Gb")
parser.add_argument("nb_gpu", help="How many GPU you need")
parser.add_argument(
    "wall_time",
    default=47,
    help="Define the maximum time of your execution in integer hour minus one",
)
args = parser.parse_args()

fichier = open("job_script.sh", "w")
fichier.write("#!/bin/bash" + "\n")

if args.cluster_type == "slurm":
    fichier.write("#SBATCH --job-name=" + args.experience_name + "\n")
    fichier.write("#SBATCH --chdir=/home/genouest/dyliss/nbuton/" + "\n")
    fichier.write(
        "#SBATCH --output=tfpc/data/outputs_jobs/output_"
        + args.experience_name
        + ".txt"
        + "\n"
    )
elif args.cluster_type == "oar" or args.cluster_type == "oar_nef":
    pass
else:
    raise RuntimeError("Cluster type unknown.")

fichier.write("source python_env_these/bin/activate" + "\n")
fichier.write("cd tfpc/" + "\n")
fichier.write(
    "python3 training.py "
    + FOLDER_CONFIG_FILE
    + args.experience_name
    + "/config.json"
    + "\n"
)
fichier.close()

# submit the job
if args.cluster_type == "slurm":
    command_slurm = (
        "sbatch --mem="
        + args.ram_qt
        + "G --gres=gpu:"
        + args.nb_gpu
        + " -p gpu job_script.sh"
    )
    os.system(command_slurm)
elif args.cluster_type == "oar":
    os.system("chmod u+x job_script.sh")
    COMMAND_OAR = (
        "oarsub  "
        + '-l {"gpu_mem >= '
        + args.ram_qt
        + ' "}/nodes=1/gpu_device='
        + args.nb_gpu
        + ",walltime="
        + args.wall_time
        + ":59:00"
        + " -O /srv/tempdd/nbuton/SCRATCH/"
        + args.experience_name
        + ".output"
        + " -E /srv/tempdd/nbuton/SCRATCH/"
        + args.experience_name
        + ".error"
        + " -n "
        + args.experience_name
        + " ./job_script.sh"
    )
    print(COMMAND_OAR)
    os.system(COMMAND_OAR)
elif args.cluster_type == "oar_nef":
    os.system("chmod u+x job_script.sh")
    COMMAND_OAR = (
        "oarsub  "
        + " -p \"gpu='YES' \" -l /nodes=1"
        + ",walltime=90:00:00"
        + " -O output_nef/"
        + args.experience_name
        + ".output"
        + " -E output_nef/"
        + args.experience_name
        + ".error"
        + " -n "
        + args.experience_name
        + " ./job_script.sh"
    )
    print(COMMAND_OAR)
    os.system(COMMAND_OAR)

else:
    raise RuntimeError("Cluster type unknown.")
